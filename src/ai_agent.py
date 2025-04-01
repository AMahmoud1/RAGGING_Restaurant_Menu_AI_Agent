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
        self.decompose_llm = Ollama(
            model=llm_model, temperature=mqr_temp, keep_alive=keep_alive
        )

        # Define llm for response generation
        self.response_llm = Ollama(
            model=llm_model, temperature=final_response_temp, keep_alive=keep_alive
        )

        # Create decompose chain
        self.decompose_chain = self.decompose_chain_creation(self.decompose_llm)

        # Create llm chain
        self.llm_chain = self.llm_chain_creation(self.response_llm)

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

    def decompose_chain_creation(self, llm):
        """
        Function to decompose the original query into multiple sub-queries.

        Arguments:
            query_text: The original query text
            decompose_chain: The decomposition chain

        Returns:
            sub_queries: The decomposed sub-queries
        """
        # Prompt for decomposing the original query into multiple sub-queries
        decompose_prompt = """
        You are an expert in retrieval-augmented generation (RAG) for restaurant AI applications. Your task is to rewrite the user's query into {num_queries} highly relevant and precise queries for optimal document retrieval.  

        Enhance the queries by incorporating relevant nutritional details such as fat content, calories, ingredients, allergens, and other health-related information, if applicable. Ensure that the queries are well-structured to retrieve data from online sources, menus, and nutritional databases.  

        Retain key terms from the original query while eliminating ambiguity to prevent incorrect retrieval. Ensure the queries stay focused on restaurant-related information without unnecessary generalization.  

        Do not return any prefix or explanations—only the queries.  

        User Query: {query}  

        Generate a list of {num_queries} refined queries in the following format:  
        1.  
        2.  
        3.  
        4.  
        """

        DECOMPOSE_PROMPT = PromptTemplate.from_template(decompose_prompt)

        # Define the decomposition chain
        decompose_chain = LLMChain(
            llm=llm, prompt=DECOMPOSE_PROMPT, callbacks=None, verbose=False
        )
        return decompose_chain

    def llm_chain_creation(self, llm):
        """
        Function to create the LLM chain for generating responses.

        Arguments:
            llm: The LLM model object

        Returns:
            llm_chain: The LLM chain object
        """
        # Prompt for generating responses based on the context and question
        prompt = """
        You are an expert in retrieval-augmented generation (RAG) for restaurant recommendations and nutrition analysis.  
        Provide a **short and precise** response to the user's query based solely on the provided context, including menu items, nutritional databases, and calorie information.  

        1. Use the following pieces of context to answer the question at the end.  
        2. **Do not include external knowledge**, only use the provided menu details, nutrition databases, and available dietary information.  
        3. **Summarize the information** clearly and reference specific dishes or ingredients when applicable.  
        4. If the context does not provide an answer, explicitly state: "I don't have enough information."  
        5. Keep the answer crisp and limited to 3-4 sentences.  
        6. Provide a concise and helpful response. If not enough information is available, explicitly state so.  

        \n\n**Context:** {context}  

        \n\n**Question:** {question}  

        **Helpful Answer:**  
        """

        QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

        llm_chain = LLMChain(
            llm=llm, prompt=QA_CHAIN_PROMPT, callbacks=None, verbose=False
        )
        return llm_chain

    def infer(self, query_text):
        """
        Function to process the original query and generate a single RAG response.

        Arguments:
            query_text (Str or List): The original query text

        Returns:
            result: The generated RAG response
        """
        # Step 1: Decompose the original query into sub-queries
        sub_queries = self.decompose_chain.run(
            query=query_text, num_queries=self.num_queries, return_only_outputs=True
        )
        sub_queries = sub_queries.strip().split(
            "\n"
        )  # Assuming sub-queries are separated by newlines
        sub_queries.append(query_text)  # Add the original query to the list

        # Step 2: Retrieve contexts for each sub-query
        all_contexts = set()  # Use a set to store unique contexts
        for sub_query in sub_queries:
            contexts = self.retrieve_contexts(sub_query)
            all_contexts.update(contexts)  # Add contexts to the set

        # Step 3: Combine all contexts into a single string
        combined_context = "\n\n".join(all_contexts)

        # Step 4: Pass the combined context and original query to the LLM
        print(f"[LLM Chain] Calling with queries: {sub_queries}")
        result = self.llm_chain.run(
            context=combined_context, question=query_text, return_only_outputs=True
        )
        print(f"[LLM Chain] Response: {result}")
        return result
