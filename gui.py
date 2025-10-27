import base64
import datetime
import json
import os
import streamlit as st
from llama_cloud import FilterOperator, MetadataFilter, MetadataFilters

from modules.ntt_rag import NTTRAG

def get_file_download_link(file_path):
    """Generate a download link for a file"""
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        filename = os.path.basename(file_path)
        return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{filename}</a>'
    except Exception as e:
        return None

def format_file_size(size_in_bytes):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.1f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.1f} GB"

def format_date(date_str):
    """Format date string to a more readable format"""
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj.strftime('%B %d, %Y')
    except:
        return date_str

if __name__ == "__main__":
    
    with open("settings.json", "r") as f:
        settings = json.load(f)

    nttrag = NTTRAG(settings["rag"])
    
    st.set_page_config(
        page_title="NTT ragIt", 
        layout="wide", #"centered", 
        initial_sidebar_state="auto", 
        #menu_items=None
        )
   
    st.info("NTT Consultants ltd.", icon="📃")
    
    st.title("ragIt™ - Shiponi's Office AI Assistant")
    
    # Add client filter in sidebar
    client_filter = st.sidebar.text_input("Filter by client:")
   
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if they exist
            if "sources" in message and message["sources"]:
                with st.expander("View Sources"):
                    for idx, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {idx}**")
                        
                        # Display metadata in a structured way
                        metadata = source.get("metadata", {})
                        if metadata:
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                if "file_name" in metadata:
                                    st.markdown(f"📄 **{metadata['file_name']}**")
                                if "file_path" in metadata:
                                    download_link = get_file_download_link(metadata["file_path"])
                                    if download_link:
                                        st.markdown(f"Download: {download_link}", unsafe_allow_html=True)
                                
                            with col2:
                                if "page_label" in metadata:
                                    st.markdown(f"Page: {metadata['page_label']}")
                                if "file_size" in metadata:
                                    st.markdown(f"Size: {format_file_size(metadata['file_size'])}")
                            
                            if "creation_date" in metadata:
                                st.markdown(f"Created: {format_date(metadata['creation_date'])}")
                            if "last_modified_date" in metadata:
                                st.markdown(f"Modified: {format_date(metadata['last_modified_date'])}")
                        
                        # Display relevance score
                        if source.get("score"):
                            st.markdown(f"Relevance Score: {source['score']}")
                        
                        # Display source text
                        st.markdown("**Extracted Text:**")
                        st.markdown(f"```\n{source['text'][:500]}...\n```")
                        st.markdown("---")

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get and display assistant response
        with st.chat_message("assistant"):
            
            metadata_filters = None
            if client_filter:
                metadata_filters = MetadataFilters( filters=[  MetadataFilter(key="client",operator=FilterOperator.EQUAL_TO, value=client_filter),  ],)

            result = nttrag.query_with_sources(prompt,metadata_filters)
            response_text = result["response"]
            sources = result["sources"]
            
            st.markdown(response_text)
            
            # Display sources in expandable section
            if sources:
                with st.expander("View Sources"):
                    for idx, source in enumerate(sources, 1):
                        st.markdown(f"**Source {idx}**")
                        
                        # Display metadata in a structured way
                        metadata = source.get("metadata", {})
                        if metadata:
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                if "file_name" in metadata:
                                    st.markdown(f"📄 **{metadata['file_name']}**")
                                if "file_path" in metadata:
                                    download_link = get_file_download_link(metadata['file_path'])
                                    if download_link:
                                        st.markdown(f"Download: {download_link}", unsafe_allow_html=True)
                                
                            with col2:
                                if "page_label" in metadata:
                                    st.markdown(f"Page: {metadata['page_label']}")
                                if "file_size" in metadata:
                                    st.markdown(f"Size: {format_file_size(metadata['file_size'])}")
                            
                            if "creation_date" in metadata:
                                st.markdown(f"Created: {format_date(metadata['creation_date'])}")
                            if "last_modified_date" in metadata:
                                st.markdown(f"Modified: {format_date(metadata['last_modified_date'])}")
                        
                        # Display relevance score
                        if source.get("score"):
                            st.markdown(f"Relevance Score: {source['score']}")
                        
                        # Display source text
                        st.markdown("**Extracted Text:**")
                        st.markdown(f"```\n{source['text'][:500]}...\n```")
                        st.markdown("---")

            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_text,
                "sources": sources
            })