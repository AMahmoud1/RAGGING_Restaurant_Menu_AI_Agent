import os
import pickle

from langchain.chains.llm import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker


class AI_Agent:
    def __init__(self, args, keep_alive=None):
        # Access the model names
        llm_model = args.llm_model
        pdf_path = args.pdf_path
        fiass_index_path = args.fiass_index_path
        faiss_k = args.faiss_k
        self.num_queries = args.num_queries
        mqr_temp = args.mqr_temp
        final_response_temp = args.final_response_temp

       # This loads only one model in Ollama with multiple configurations from Python
        # refer to: https://github.com/ollama/ollama/issues/9054?utm_source=chatgpt.com
        # https://stackoverflow.com/questions/79526074/ollama-model-keep-in-memory-and-prevent-unloading-between-requests-keep-alive
        # Define llm for decomposition
        self.decompose_llm = Ollama(model=llm_model, temperature=mqr_temp, keep_alive=keep_alive)

        # Define llm for response generation
        self.response_llm = Ollama(model=llm_model, temperature=final_response_temp, keep_alive=keep_alive)

    def infer(self, query_text):
        pass