import logging
import os
import requests
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from chroma_vec import create_vector_store

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_local_chatbot(model_name="llama3.1:8b"):
    BASE_DIR = "/Users/issacleung/Desktop/RAGpipline"
    logging.info(f"Loading vector store for model: {model_name}")

    try:
        vector_db = create_vector_store()
        if vector_db is None:
            raise ValueError("Vector store creation returned None")
    except Exception as e:
        logging.error(f"Failed to load vector store: {str(e)}")
        raise Exception(f"Failed to load vector store: {str(e)}")

    # Only load Parents prompt
    parents_prompt_file = os.path.join(BASE_DIR, "parents_prompt.txt")
    
    default_parents_prompt = (
        "You are a compassionate and knowledgeable parenting assistant designed to provide practical, "
        "flexible, and holistic strategies for parents. Your responses should be tailored to the "
        "parent's specific needs, offering guidance on supporting their child with ADHD."
    )

    if os.path.exists(parents_prompt_file) and os.path.getsize(parents_prompt_file) > 0:
        with open(parents_prompt_file, "r", encoding="utf-8") as f:
            parents_prompt = f.read().strip()
    else:
        parents_prompt = default_parents_prompt

    # Ollama check
    try:
        response = requests.get("http://127.0.0.1:11434", timeout=5)
        if response.status_code != 200:
            raise Exception("Ollama server not responding")
    except Exception as e:
        raise Exception("Ollama server is not running. Please run 'ollama serve' first.")

    llm = ChatOllama(model=model_name, temperature=0.5, request_timeout=120)

    # Original-style retriever (kept close to your original)
    primary_retriever = vector_db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.1}
    )
    fallback_retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    parents_rag_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(parents_prompt),
        HumanMessagePromptTemplate.from_template("Context: {context}\n\nQuery: {query}")
    ])

    def format_docs(docs):
        if not docs:
            return ""
        return "\n\n".join([
            f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}"
            for doc in docs
        ])

    parents_chain = (
        {
            "context": primary_retriever | format_docs,
            "query": RunnablePassthrough()
        }
        | parents_rag_prompt
        | llm
        | StrOutputParser()
    )

    def query_parent(query):
        try:
            augmented_query = f"{query.strip().lower()} bullying ADHD"
            logging.info(f"Processing augmented query: {augmented_query}")

            try:
                result = parents_chain.invoke(augmented_query)
                response = result if isinstance(result, str) else str(result)
            except Exception as e:
                logging.warning(f"Primary chain failed: {str(e)}. Trying fallback.")
                try:
                    context = format_docs(fallback_retriever.invoke(augmented_query))
                    prompt_input = {"context": context, "query": augmented_query}
                    response = (parents_rag_prompt | llm | StrOutputParser()).invoke(prompt_input)
                except Exception as e:
                    logging.warning(f"Fallback failed. Using direct LLM.")
                    fallback_prompt = ChatPromptTemplate.from_messages([
                        SystemMessagePromptTemplate.from_template(parents_prompt),
                        HumanMessagePromptTemplate.from_template("{query}")
                    ])
                    fallback_chain = fallback_prompt | llm | StrOutputParser()
                    response = fallback_chain.invoke(augmented_query)

            return response.strip()
        except Exception as e:
            logging.error(f"Error querying Ollama: {str(e)}")
            return f"Error: {str(e)}"

    logging.info("Parent-only RAG setup complete")
    return None, parents_chain, query_parent, None, parents_prompt


if __name__ == "__main__":
    setup_local_chatbot()
    print("Backend ready (Parent Mode)")