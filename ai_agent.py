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

        # 1. Load document
        self.docs = self.load_documents(path=pdf_path)

        # Split sentences into chunks semantically
        self.documents = self.semantic_chunker(self.docs)

        # Create Vector db embeddings for each text chunk​
        self.retriever = self.retriever_creation(
            self.documents, fiass_index_path, faiss_k
        )

        # This loads only one model in Ollama with multiple configurations from Python
        # refer to: https://github.com/ollama/ollama/issues/9054?utm_source=chatgpt.com
        # https://stackoverflow.com/questions/79526074/ollama-model-keep-in-memory-and-prevent-unloading-between-requests-keep-alive
        # Define llm for decomposition
        self.decompose_llm = Ollama(model=llm_model, temperature=mqr_temp, keep_alive=keep_alive)

        # Define llm for response generation
        self.response_llm = Ollama(model=llm_model, temperature=final_response_temp, keep_alive=keep_alive)

    def load_documents(self, path):
        """
        Load the PDF document and return the number of pages.

        Arguments:
            path (str): The path to the PDF document

        Returns:
            docs: The loaded PDF document
        """
        loader = PDFPlumberLoader(path)
        docs = loader.load()

        # Check the number of pages
        print("Number of pages in the PDF:", len(docs))
        return docs

    def semantic_chunker(self, docs):
        """
        Split the documents into chunks using the SemanticChunker.

        Arguments:
            docs: The loaded PDF document

        Returns:
            documents: The split documents
        """
        text_splitter = SemanticChunker(HuggingFaceEmbeddings())
        documents = text_splitter.split_documents(docs)

        print("Number of chunks created: ", len(documents))
        return documents

    def retriever_creation(self, documents, fiass_index_path, faiss_k):
        """
        Create a retriever using the FAISS index database.

        Arguments:
            documents: The split documents
            fiass_index_path: The path to the FAISS index database

        Returns:
            retriever: The retriever object
        """
        # Check if the FAISS index file exists
        if os.path.exists(fiass_index_path):
            print(f"Loading existing FAISS index from {fiass_index_path}")
            with open(fiass_index_path, "rb") as f:
                vector = pickle.load(f)
        else:
            print(f"Creating new FAISS index and saving to {fiass_index_path}")
            # Instantiate the embedding model
            embedder = HuggingFaceEmbeddings()
            # Create the vector store
            vector = FAISS.from_documents(documents, embedder)
            # Save the FAISS index to a file
            with open(fiass_index_path, "wb") as f:
                pickle.dump(vector, f)

        retriever = vector.as_retriever(
            search_type="similarity", search_kwargs={"k": faiss_k}
        )
        return retriever

    def retrieve_contexts(self, query):
        """
        Retrieve the relevant documents for a given query.

        Arguments:
            query: The query text

        Returns:
            contexts: The retrieved contexts
        """
        retrieved_docs = self.retriever.get_relevant_documents(query)
        contexts = [doc.page_content for doc in retrieved_docs]
        return contexts

    def infer(self, query_text):
        pass