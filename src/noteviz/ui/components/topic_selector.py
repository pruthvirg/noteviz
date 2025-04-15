"""Topic selector component for NoteViz."""
import streamlit as st

def topic_selector(topics, descriptions):
    """Render the topic selector component."""
    st.subheader("Choose a Topic")
    
    selected_topic = None
    
    # Display topics in a grid using columns
    cols = st.columns(2)
    for idx, topic in enumerate(topics):
        with cols[idx % 2]:
            with st.container():
                st.markdown(f"### {topic}")
                st.markdown(descriptions.get(topic, ""))
                if st.button("Select", key=f"btn_{topic}", use_container_width=True):
                    selected_topic = topic
    
    # Custom topic input
    st.subheader("Or Enter Your Own Topic")
    custom_topic = st.text_input("Enter a topic", value=st.session_state.get('custom_topic', ''))
    if st.button("Visualize Custom Topic", use_container_width=True):
        if custom_topic:
            selected_topic = custom_topic
            st.session_state.custom_topic = custom_topic
    
    return selected_topic 