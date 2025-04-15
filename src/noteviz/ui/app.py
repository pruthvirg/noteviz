"""Main Streamlit app for NoteViz."""
import streamlit as st
import os
import logging
from pathlib import Path
import tempfile
from noteviz.core.pdf import PDFConfig, PageAwarePDFProcessor, PageAwareChunk
from noteviz.core.llm.openai import OpenAITopicExtractor, OpenAISummarizer
from noteviz.core.llm import TopicExtractorConfig, SummarizerConfig
from noteviz.core.flowchart.openai import OpenAIFlowchartGenerator
from noteviz.config.test_config import TEST_MODE, TEST_RESPONSES
from pypdf import PdfReader

# Import UI components
from noteviz.ui.components.pdf_uploader import pdf_uploader
from noteviz.ui.components.topic_selector import topic_selector
from noteviz.ui.components.flowchart_viewer import flowchart_viewer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config - MUST be the first Streamlit command
st.set_page_config(page_title="NoteViz", page_icon="📚", layout="wide")

# Add custom CSS
st.markdown("""
    <style>
        /* Main Title */
        h1.main-title {
            color: #1f1f1f;
            font-size: 3rem;
            font-weight: 800;
            margin: 1rem 0;
            padding-bottom: 0.8rem;
            position: relative;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .logo-icon {
            color: #4CAF50;
            font-size: 2.8rem;
            margin-right: 0.2rem;
        }
        .title-text {
            background: linear-gradient(90deg, #1f1f1f 0%, #2e2e2e 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        h1.main-title::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 120px;
            height: 6px;
            background: linear-gradient(90deg, #4CAF50, #45a049);
            border-radius: 3px;
        }
        
        /* Section Headers */
        h2.section-header {
            color: #1f1f1f;
            font-size: 2rem;
            font-weight: 700;
            margin: 1rem 0 0 0;
            padding-bottom: 0.5rem;
            position: relative;
        }
        h2.section-header::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 60px;
            height: 4px;
            background: linear-gradient(90deg, #4CAF50, #45a049);
            border-radius: 2px;
        }
        
        /* Topic Container and Boxes */
        .topic-container {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
            width: 100%;
            margin: 0;
            padding: 0.5rem;
        }
        .topic-box {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 12px;
            padding: 1.8rem;
            cursor: pointer;
            transition: all 0.2s ease-in-out;
            width: 100%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            position: relative;
            overflow: hidden;
            margin-bottom: 1rem;
        }
        .topic-box:hover {
            background-color: #ffffff;
            transform: translateY(-4px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            border-color: #4CAF50;
        }
        .topic-box:active {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .topic-box h3 {
            margin: 0;
            color: #262730;
            font-size: 1.3rem;
            font-weight: 600;
        }
        .topic-box p {
            margin: 1rem 0 0 0;
            color: #666666;
            font-size: 1rem;
            line-height: 1.5;
        }
        .topic-box::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, #4CAF50, #45a049);
            transform: scaleX(0);
            transition: transform 0.3s ease;
            transform-origin: left;
        }
        .topic-box:hover::after {
            transform: scaleX(1);
        }

        /* Streamlit specific overrides */
        .stMarkdown {
            cursor: pointer;
        }
        .stMarkdown > div {
            margin-bottom: 0 !important;
        }
        [data-testid="column"] {
            padding: 0.3rem !important;
        }
        /* Hide Streamlit's default elements */
        [data-testid="stMarkdown"] {
            width: 100%;
            margin-bottom: 0 !important;
        }
        div[data-testid="stMarkdown"] {
            margin-bottom: 0 !important;
        }
        div[data-testid="stMarkdown"] > div {
            margin-bottom: 0 !important;
        }
        div[data-testid="stVerticalBlock"] {
            gap: 0 !important;
            padding: 0 !important;
        }
        section[data-testid="stSidebar"] {
            padding-top: 0 !important;
        }
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 0 !important;
            max-width: 100% !important;
        }
        .element-container {
            margin-bottom: 0 !important;
        }
        /* Custom styling for other headers */
        .custom-header {
            color: #1f1f1f;
            font-size: 1.5rem;
            font-weight: 600;
            margin: 1.5rem 0 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = None
if 'topics' not in st.session_state:
    st.session_state.topics = []
if 'selected_topic' not in st.session_state:
    st.session_state.selected_topic = None
if 'page_aware_chunks' not in st.session_state:
    st.session_state.page_aware_chunks = None

# Set up the page
st.markdown("""
    <h1 class="main-title">
        <span class="logo-icon">📚</span>
        <span class="title-text">NoteViz - Visualize your Notes</span>
    </h1>
""", unsafe_allow_html=True)

# Show test mode indicator
if TEST_MODE:
    st.info("🧪 Test Mode Enabled")

# Check API key
if not TEST_MODE and not os.getenv("OPENAI_API_KEY"):
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

@st.cache_data
def process_pdf(pdf_content):
    """Process a PDF file and extract topics."""
    logger.info("Starting PDF processing...")
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_content)
        pdf_path = Path(tmp_file.name)
        logger.info(f"Created temporary file: {pdf_path}")
    
    try:
        # Initialize components
        config = PDFConfig()
        processor = PageAwarePDFProcessor(config)
        topic_extractor = OpenAITopicExtractor(TopicExtractorConfig())
        
        # Process PDF
        logger.info("Processing PDF...")
        chunks = []
        
        # Use context manager to ensure file is closed
        with open(pdf_path, 'rb') as pdf_file:
            reader = PdfReader(pdf_file)
            logger.info(f"PDF loaded. Total pages: {len(reader.pages)}")
            
            for page_num, page in enumerate(reader.pages, 1):
                logger.info(f"Processing page {page_num}...")
                page_text = page.extract_text()
                if not page_text.strip():
                    continue
                
                chunk = PageAwareChunk(
                    text=page_text,
                    page_number=page_num,
                    start_char=0,
                    end_char=len(page_text)
                )
                chunks.append(chunk)
                logger.info(f"Added chunk for page {page_num}")
        
        # Extract topics
        logger.info("Extracting topics...")
        if TEST_MODE:
            topics = TEST_RESPONSES["topics"]
            descriptions = TEST_RESPONSES["topic_descriptions"]
            logger.info("Using test topics")
        else:
            chunk_texts = [chunk.text for chunk in chunks]
            topics = topic_extractor.extract_topics_sync(chunk_texts)
            descriptions = {topic: f"Description for {topic}" for topic in topics}
            logger.info(f"Extracted {len(topics)} topics")
        
        return chunks, topics, descriptions
    finally:
        # Clean up temporary file - now safe to delete since file handle is closed
        try:
            os.unlink(pdf_path)
            logger.info("Cleaned up temporary file")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary file: {e}")
            # Don't raise the error as it's not critical

# File uploader
uploaded_file = st.file_uploader("Upload your PDF", type=['pdf'])

if uploaded_file:
    # Process PDF when file is uploaded
    with st.spinner("Analyzing document..."):
        try:
            chunks, topics, descriptions = process_pdf(uploaded_file.getvalue())
            st.session_state.page_aware_chunks = chunks
            st.session_state.topics = topics
            st.session_state.topic_descriptions = descriptions
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            st.error(f"Error processing document: {str(e)}")
            st.stop()
    
    # Display topics in a grid
    st.markdown('<h2 class="section-header">✨ Explore Topics in Your Document</h2>', unsafe_allow_html=True)
    
    # Create columns for the grid layout
    col1, col2 = st.columns(2)
    
    # Display topics in columns
    for idx, topic in enumerate(topics):
        with col1 if idx % 2 == 0 else col2:
            st.markdown(f"""
                <style>
                    div[data-testid="stButton"] > button {{
                        width: 100%;
                        background-color: #f8f9fa;
                        border: 1px solid #e9ecef;
                        border-radius: 12px;
                        padding: 1.8rem;
                        height: auto;
                        text-align: left;
                        color: inherit;
                        margin-bottom: 1.2rem;
                    }}
                    div[data-testid="stButton"] > button:hover {{
                        background-color: #ffffff;
                        border-color: #4CAF50;
                        transform: translateY(-4px);
                        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
                    }}
                    div[data-testid="stButton"] > button:active {{
                        transform: translateY(-2px);
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    }}
                    div[data-testid="stButton"] > button p {{
                        font-size: 1rem;
                        color: #666666;
                        margin-top: 0.8rem;
                    }}
                    div[data-testid="stButton"] > button strong {{
                        display: block;
                        font-size: 1.4rem;
                        font-weight: 800;
                        color: #1f1f1f;
                        margin-bottom: 0.8rem;
                        letter-spacing: -0.02em;
                    }}
                </style>
            """, unsafe_allow_html=True)
            
            if st.button(
                f"""**{topic}**

                {descriptions.get(topic, '')}""",
                key=f"topic_{idx}"
            ):
                st.session_state.selected_topic = topic
    
    # Custom topic input
    st.markdown('<h3 class="custom-header">✨ Or Enter Your Own Topic</h3>', unsafe_allow_html=True)
    custom_topic = st.text_input("Enter a topic", key="custom_topic")
    if st.button("Visualize", use_container_width=True) and custom_topic:
        st.session_state.selected_topic = custom_topic
    
    # Show flowchart and summary if a topic is selected
    if st.session_state.selected_topic:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Flowchart: {st.session_state.selected_topic}")
            with st.spinner("Generating flowchart..."):
                try:
                    if TEST_MODE:
                        from noteviz.core.flowchart.base import Flowchart, Node, Edge
                        nodes = [
                            Node(id="A", label="Introduction to AI", confidence=0.9),
                            Node(id="B", label="Machine Learning Basics", confidence=0.8),
                            Node(id="C", label="Neural Networks", confidence=0.7),
                            Node(id="D", label="Deep Learning Applications", confidence=0.8)
                        ]
                        edges = [
                            Edge(source="A", target="B", label="includes"),
                            Edge(source="B", target="C", label="includes"),
                            Edge(source="C", target="D", label="includes")
                        ]
                        flowchart = Flowchart(
                            title=f"Flowchart for {st.session_state.selected_topic}",
                            description="Test flowchart",
                            nodes=nodes,
                            edges=edges
                        )
                    else:
                        generator = OpenAIFlowchartGenerator()
                        flowchart = generator.generate_flowchart_sync(
                            st.session_state.page_aware_chunks,
                            st.session_state.selected_topic
                        )
                    
                    # Render flowchart
                    if isinstance(flowchart, str):
                        mermaid_code = flowchart
                    else:
                        from noteviz.core.flowchart.mermaid import MermaidRenderer
                        renderer = MermaidRenderer()
                        mermaid_code = renderer.render(flowchart)
                    
                    st.components.v1.html(
                        f"""
                        <div class="mermaid">
                        {mermaid_code}
                        </div>
                        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
                        <script>
                            mermaid.initialize({{ 
                                startOnLoad: true, 
                                theme: 'default',
                                flowchart: {{
                                    htmlLabels: true,
                                    curve: 'basis'
                                }}
                            }});
                        </script>
                        """,
                        height=500
                    )
                except Exception as e:
                    logger.error(f"Error generating flowchart: {str(e)}")
                    st.error(f"Error generating flowchart: {str(e)}")
        
        with col2:
            st.subheader("Summary")
            with st.spinner("Generating summary..."):
                try:
                    if TEST_MODE:
                        summary = TEST_RESPONSES["topic_summaries"].get(
                            st.session_state.selected_topic,
                            "Test summary"
                        )
                    else:
                        summarizer = OpenAISummarizer(SummarizerConfig())
                        summary = summarizer.generate_summary_sync(
                            st.session_state.page_aware_chunks,
                            st.session_state.selected_topic
                        )
                    st.write(summary)
                except Exception as e:
                    logger.error(f"Error generating summary: {str(e)}")
                    st.error(f"Error generating summary: {str(e)}")
else:
    st.info("Upload your PDF to see the topics!") 