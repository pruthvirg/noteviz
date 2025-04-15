"""Flowchart viewer component for NoteViz."""
import streamlit as st
import streamlit.components.v1 as components

def render_mermaid(flowchart):
    """Render a Mermaid flowchart."""
    # Create a unique ID for this flowchart
    flowchart_id = f"flowchart_{id(flowchart)}"
    
    # Get the Mermaid code
    if isinstance(flowchart, str):
        mermaid_code = flowchart
    else:
        from noteviz.core.flowchart.mermaid import MermaidRenderer
        renderer = MermaidRenderer()
        mermaid_code = renderer.render(flowchart)
    
    # Render the flowchart
    components.html(
        f"""
        <div id="{flowchart_id}" class="mermaid">
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

def flowchart_viewer(flowchart, topic):
    """Render the flowchart viewer component."""
    if flowchart:
        st.subheader(f"Flowchart: {topic}")
        render_mermaid(flowchart) 