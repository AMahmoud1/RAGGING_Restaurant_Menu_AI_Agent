import argparse
import os

import streamlit as st
import torch

from ai_agent import AI_Agent

st.set_page_config(page_title="Restaurant AI Agent", page_icon="📖")
st.title("Talk to an AI Agent about your favorite meal 🍔🍕🍣")


def arguments_parser():
    """
    Parse the command-line arguments

    Arguments:
        None

    Returns:
        args: The parsed arguments
    """
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process model names")
    parser.add_argument(
        "llm_model", type=str, help="LLM model name (e.g., deepseek-r1:7b)"
    )
    parser.add_argument("pdf_path", type=str, help="Path to Esbaar Handbook PDF")
    parser.add_argument(
        "fiass_index_path", type=str, help="Path to FIASS index database"
    )
    parser.add_argument("faiss_k", type=int, help="Number of chunks returned by FAISS")
    parser.add_argument(
        "num_queries", type=int, help="Number of chunks returned by FAISS"
    )
    parser.add_argument("chunks_files_path", type=str, help="Path to Chunks")
    parser.add_argument("mqr_temp", type=float, help="Multi-Query Ragging Temparture")
    parser.add_argument(
        "final_response_temp", type=float, help="Final Response from LLM Temparture"
    )

    # Parse the arguments
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Bug fix: https://github.com/VikParuchuri/marker/issues/442#issuecomment-2636393925
    torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

    args = arguments_parser()

    # Initialize AI Agent once and store in session state
    if "ai_agent" not in st.session_state:
        st.session_state.ai_agent = AI_Agent(args)

    # User input prompt
    user_prompt = st.text_area("Enter your prompt:", "")

    if st.button("Submit"):
        if user_prompt.strip():
            # Use the AI agent from session state
            response = st.session_state.ai_agent.infer(user_prompt)
            st.subheader("AI Response:")
            st.write(response)
        else:
            st.warning("Please enter a prompt before submitting.")
