"""PDF uploader component for NoteViz."""
import streamlit as st
from pathlib import Path
import tempfile
import os

def pdf_uploader():
    """Render the PDF uploader component."""
    uploaded_file = st.file_uploader("Upload your PDF", type=['pdf'])
    
    if uploaded_file:
        if uploaded_file != st.session_state.pdf_path:
            st.session_state.pdf_path = uploaded_file
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                st.session_state.temp_pdf_path = Path(tmp_file.name)
            return True
    else:
        st.info("Upload your PDF to see the topics!")
        return False 