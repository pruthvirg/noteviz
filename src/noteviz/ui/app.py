"""Main Streamlit app for NoteViz."""
import streamlit as st
import tempfile
import os
from pathlib import Path
import asyncio
from noteviz.core.pdf import PDFConfig, PyPDFProcessor
from noteviz.core.llm.openai import OpenAITopicExtractor, OpenAISummarizer
from noteviz.core.llm import TopicExtractorConfig, SummarizerConfig
from noteviz.core.flowchart.openai import OpenAIFlowchartGenerator
from noteviz.core.flowchart.mermaid import MermaidRenderer
import streamlit.components.v1 as components

# Set page config with a beautiful theme
st.set_page_config(
    page_title="NoteViz",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for a beautiful UI
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        background-color: #f5f7f9;
    }
    .stButton>button {
        background-color: #4e54c8;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #3a3f9e;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .upload-header {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 22px;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .topic-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        cursor: pointer;
        display: inline-block;
        width: calc(50% - 10px);
        margin-right: 10px;
        vertical-align: top;
    }
    .topic-card:hover {
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .topic-card.selected {
        border: 2px solid #4e54c8;
        background-color: #f8f9ff;
    }
    .topic-card h3 {
        font-size: 16px;
        margin-bottom: 5px;
    }
    .topic-card p {
        font-size: 12px;
        margin-bottom: 5px;
    }
    .flowchart-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .summary-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .custom-topic-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .node-tooltip {
        position: absolute;
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        max-width: 300px;
        z-index: 1000;
        display: none;
    }
    .loading-message {
        color: #4e54c8;
        font-weight: bold;
        margin: 10px 0;
    }
    .topic-grid {
        display: flex;
        flex-wrap: wrap;
        margin: 0 -5px;
    }
    .topic-grid-item {
        flex: 0 0 calc(50% - 10px);
        margin: 5px;
    }
    .stButton>button {
        width: 100%;
        background-color: #4e54c8;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #3a3f9e;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
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
if 'custom_topic' not in st.session_state:
    st.session_state.custom_topic = ""
if 'flowchart' not in st.session_state:
    st.session_state.flowchart = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'node_click' not in st.session_state:
    st.session_state.node_click = None
if 'error' not in st.session_state:
    st.session_state.error = None
if 'summary_timeout' not in st.session_state:
    st.session_state.summary_timeout = False
if 'summary_progress' not in st.session_state:
    st.session_state.summary_progress = None
if 'summary_retries' not in st.session_state:
    st.session_state.summary_retries = 0

async def process_pdf(pdf_path):
    """Process the PDF and extract topics."""
    try:
        # Initialize components
        config = PDFConfig()
        processor = PyPDFProcessor(config)
        
        # Initialize topic extractor with config
        topic_config = TopicExtractorConfig(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000,
            num_topics=5
        )
        topic_extractor = OpenAITopicExtractor(topic_config)
        
        # Process PDF
        chunks = await processor.process_pdf(pdf_path)
        
        # Extract topics
        topics = await topic_extractor.extract_topics(chunks)
        
        return topics
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return []

async def generate_flowchart(pdf_path, topic, keywords=None):
    """Generate a flowchart for the given topic."""
    try:
        # Initialize components
        config = PDFConfig()
        processor = PyPDFProcessor(config)
        flowchart_generator = OpenAIFlowchartGenerator()
        
        # Process PDF
        chunks = await processor.process_pdf(pdf_path)
        text = "\n".join(chunks)  # Combine chunks into a single text
        
        # Generate flowchart
        flowchart = await flowchart_generator.generate_flowchart(
            text=text,
            topic=topic,
            keywords=keywords or []
        )
        
        return flowchart
    except Exception as e:
        st.error(f"Error generating flowchart: {str(e)}")
        return None

async def generate_summary(text: str, topic: str) -> str:
    """Generate a summary of the text using OpenAI."""
    try:
        # Initialize summarizer with config
        summarizer_config = SummarizerConfig(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=500
        )
        summarizer = OpenAISummarizer(summarizer_config)
        
        # Combine PDF chunks into a single text
        combined_text = "\n\n".join(text)
        
        # Add context about the topic
        text_with_context = f"Topic: {topic}\n\n{combined_text}"
        
        # Get summary from OpenAI
        summary = await summarizer.summarize(text=text_with_context)
        
        # Display raw API response for debugging
        st.write("Raw API Response:")
        st.code(summary, language="text")
        
        if not summary:
            st.error("Summary generation failed: Empty response from API")
            return None
            
        return summary
        
    except Exception as e:
        st.error(f"Error in summary generation: {str(e)}")
        return None

def render_mermaid(flowchart):
    """Render the flowchart using Mermaid."""
    renderer = MermaidRenderer()
    mermaid_code = renderer.render(flowchart)
    
    # Create a unique ID for this flowchart
    flowchart_id = f"flowchart_{id(flowchart)}"
    
    # Create tooltip container
    tooltip_html = """
    <div id="node-tooltip" class="node-tooltip">
        <h4 style="color: #2c3e50; margin-bottom: 8px;"></h4>
        <p style="color: #7f8c8d; font-size: 14px;"></p>
    </div>
    """
    
    # Use Streamlit's components to render Mermaid with hover and click handling
    components.html(
        f"""
        {tooltip_html}
        <div id="{flowchart_id}" class="mermaid">
        {mermaid_code}
        </div>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>
            // Initialize mermaid
            mermaid.initialize({{ 
                startOnLoad: true, 
                theme: 'default',
                flowchart: {{
                    htmlLabels: true,
                    curve: 'basis'
                }}
            }});
            
            // Function to handle node clicks
            function handleNodeClick(nodeId) {{
                window.parent.postMessage({{
                    type: 'node_click',
                    nodeId: nodeId
                }}, '*');
            }}
            
            // Function to show tooltip
            function showTooltip(node, event) {{
                const tooltip = document.getElementById('node-tooltip');
                const title = tooltip.querySelector('h4');
                const description = tooltip.querySelector('p');
                
                title.textContent = node.label;
                description.textContent = node.description || 'Click to explore this topic';
                
                tooltip.style.display = 'block';
                tooltip.style.left = event.pageX + 10 + 'px';
                tooltip.style.top = event.pageY + 10 + 'px';
            }}
            
            // Function to hide tooltip
            function hideTooltip() {{
                const tooltip = document.getElementById('node-tooltip');
                tooltip.style.display = 'none';
            }}
            
            // Add event listeners after mermaid renders
            document.addEventListener('DOMContentLoaded', function() {{
                const observer = new MutationObserver(function(mutations) {{
                    mutations.forEach(function(mutation) {{
                        if (mutation.addedNodes.length) {{
                            const nodes = document.querySelectorAll('.node');
                            nodes.forEach(node => {{
                                node.style.cursor = 'pointer';
                                
                                // Add hover events
                                node.addEventListener('mouseenter', function(e) {{
                                    const nodeId = this.id;
                                    const nodeData = {{
                                        label: this.querySelector('text').textContent,
                                        description: this.getAttribute('title') || ''
                                    }};
                                    showTooltip(nodeData, e);
                                }});
                                
                                node.addEventListener('mouseleave', hideTooltip);
                                
                                // Add click event
                                node.onclick = function() {{
                                    const nodeId = this.id;
                                    handleNodeClick(nodeId);
                                }};
                            }});
                        }}
                    }});
                }});
                
                observer.observe(document.getElementById('{flowchart_id}'), {{
                    childList: true,
                    subtree: true
                }});
            }});
        </script>
        """,
        height=500
    )
    
    # Store the flowchart in session state for reference
    if 'flowcharts' not in st.session_state:
        st.session_state.flowcharts = {}
    st.session_state.flowcharts[flowchart_id] = flowchart

async def main():
    st.title("📚 NoteViz - Visualize Your Notes")
    
    # Initialize session state
    if 'flowchart' not in st.session_state:
        st.session_state.flowchart = None
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'error' not in st.session_state:
        st.session_state.error = None
    if 'summary_timeout' not in st.session_state:
        st.session_state.summary_timeout = False
        
    # Create two columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📄 Upload Your Notes")
        uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
        
        if uploaded_file:
            # Save the uploaded file
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Extract topics
            with st.spinner("🔍 Analyzing document..."):
                topics = await process_pdf("temp.pdf")
                
            if topics:
                st.markdown("### 🎯 Select a Topic")
                # Display topics in a grid
                cols = st.columns(2)
                for i, topic in enumerate(topics):
                    with cols[i % 2]:
                        # Get topic name from Topic object
                        topic_name = topic.name if hasattr(topic, 'name') else str(topic)
                        if st.button(topic_name, key=f"topic_{i}", use_container_width=True):
                            st.session_state.selected_topic = topic_name
                            # Reset states when new topic is selected
                            st.session_state.flowchart = None
                            st.session_state.summary = None
                            st.session_state.error = None
                            st.session_state.summary_timeout = False
                
                # Custom topic input
                st.markdown("### ✍️ Or Enter Your Own Topic")
                custom_topic = st.text_input("Enter a specific topic to explore", key="custom_topic")
                if custom_topic and st.button("Generate for Custom Topic", key="custom_topic_btn"):
                    st.session_state.selected_topic = custom_topic
                    # Reset states when new topic is selected
                    st.session_state.flowchart = None
                    st.session_state.summary = None
                    st.session_state.error = None
                    st.session_state.summary_timeout = False
    
    with col2:
        if 'selected_topic' in st.session_state:
            st.markdown(f"### 📊 Visualization for: {st.session_state.selected_topic}")
            
            # Show loading message
            with st.spinner("🔄 Generating visualization..."):
                try:
                    # Generate flowchart
                    flowchart = await generate_flowchart("temp.pdf", st.session_state.selected_topic)
                    if flowchart:
                        st.session_state.flowchart = flowchart
                        st.markdown("### 📝 Generated Flowchart")
                        # Display raw flowchart data for debugging
                        st.write("Raw Flowchart Data:")
                        st.code(flowchart, language="mermaid")
                        # Render the flowchart
                        render_mermaid(flowchart)
                    else:
                        st.error("Failed to generate flowchart")
                        
                    # Generate summary with timeout
                    with st.spinner("📝 Generating summary..."):
                        try:
                            # Process PDF first
                            config = PDFConfig()
                            processor = PyPDFProcessor(config)
                            chunks = await processor.process_pdf("temp.pdf")
                            
                            # Generate summary with timeout
                            summary = await asyncio.wait_for(
                                generate_summary(chunks, st.session_state.selected_topic),
                                timeout=30.0
                            )
                            
                            if summary:
                                st.session_state.summary = summary
                                st.markdown("### 📋 Summary")
                                st.write(summary)
                            else:
                                st.error("Failed to generate summary")
                                
                        except asyncio.TimeoutError:
                            st.session_state.summary_timeout = True
                            st.error("Summary generation is taking longer than expected. Please wait...")
                        except Exception as e:
                            st.error(f"Error generating summary: {str(e)}")
                            
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.error = str(e)
            
            # Display error if any
            if st.session_state.error:
                st.error(f"An error occurred: {st.session_state.error}")
                
            # Display timeout message if applicable
            if st.session_state.summary_timeout:
                st.warning("Summary generation is still in progress. This might take a few minutes.")

if __name__ == "__main__":
    asyncio.run(main()) 