# ADHD Parent Support Chatbot - RAG Pipeline

This repository contains a **Retrieval-Augmented Generation (RAG)** chatbot designed to assist parents of children with ADHD. The system provides practical, empathetic, and evidence-informed parenting strategies using local large language models.

## Project Background

This project was developed as part of academic research at Hong Kong Baptist University. The chatbot uses a local RAG pipeline to retrieve relevant information from ADHD parenting resources and generate responses using various open-source LLMs via Ollama.

**Note on Code Refinement**:  
The submitted code is a **cleaned and refined version** of the original implementation. The core RAG logic (vector retrieval, prompt engineering, and LLM integration) remains consistent with the version used during development and result recording. Non-essential features (virtual pet, music therapy, timer, dark mode, voice input, and Student mode) have been removed to improve clarity and usability for academic review.

## Key Features

- Parent-mode focused counseling chatbot
- Supports 9 different local LLMs via Ollama
- Uses Chroma vector database with Nomic embeddings
- Clean and simple PyQt5 graphical interface
- Fully local (no external API calls required)

## Project Structure

| File | Purpose |
|------|---------|
| `ver2_test5_std.py` | **Core RAG backend**. Handles vector store loading, document retrieval, prompt construction, and LLM querying. Contains the main `setup_local_chatbot()` and `query_parent()` functions. |
| `parent_rag_chatbot.py` | **Graphical User Interface (GUI)**. Provides a clean PyQt5 interface for interacting with the chatbot. Supports model switching and displays responses in real time. |
| `chroma_vec.py` | **Vector store creation**. Converts text documents into embeddings and stores them in a persistent Chroma database (`chroma_db/`). |
| `pdf_to_text.py` | **Data preprocessing**. Converts PDF files (ADHD parenting resources) into plain text files for vectorization. |
| `parents_prompt.txt` | **System prompt** for the parent counseling mode. Defines the behavior and tone of the AI assistant. |

## Tech Stack

- **LLM Inference**: Ollama (local models)
- **Embeddings**: Nomic Embed (`nomic-ai/nomic-embed-text-v1.5`)
- **Vector Database**: Chroma
- **Framework**: LangChain
- **GUI**: PyQt5
- **Language**: Python

## How to Run

### 1. Setup Environment

```bash
conda create -n RAG python=3.11
conda activate RAG
pip install -r requirements.txt