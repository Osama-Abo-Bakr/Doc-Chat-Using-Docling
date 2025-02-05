# RAG QA System for PDF Using Docling 🤖

## Overview
The **RAG QA System for PDF Using Docling** is an AI-powered application designed to process PDFs, convert them into Markdown format, generate summaries, and provide intelligent question-answering capabilities. It utilizes advanced technologies such as **Docling**, **LangChain**, and **FAISS** to enhance document comprehension.

## Features
- 🔹 PDF Upload & Processing
- 🔹 Document Conversion to Markdown
- 🔹 Summarization using AI Models
- 🔹 Conversational QA System
- 🔹 Downloadable Summaries in Markdown Format

## Installation

### 1. Clone the Repository
```bash
git clone <repository_url>
cd <repository_folder>
```

### 2. Setting Up the Environment
You can use either **Conda** or **Python Virtual Environment (venv)** to manage dependencies.

#### Using Conda
```bash
conda create -n rag_qa_env python=3.10
conda activate rag_qa_env
pip install -r requirements.txt
```

#### Using Python venv
```bash
python -m venv rag_qa_env
source rag_qa_env/bin/activate   # On Windows use: rag_qa_env\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file based on `.env.example` and fill in the necessary credentials.
```bash
cp .env.example .env
```

## Usage
Run the application using Streamlit:
```bash
streamlit run app.py
```

- Upload your PDF, Images from the sidebar.
- Ask questions in the chat interface.
- View and download summaries in Markdown format.

## Project Structure
```
├── app.py                # Main application file
├── requirements.txt      # Dependencies
├── .env.example          # Example environment configuration
└── README.md             # Project documentation
```

## Dependencies
The project relies on the following key libraries:
- **Docling** for document conversion
- **LangChain** for building QA chains
- **FAISS** for efficient vector storage
- **Streamlit** for the interactive UI
- **Groq** for AI model integration

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests.

## License
This project is licensed under the [MIT License](LICENSE).

## Contact
For questions or support, please reach out to the project maintainers.