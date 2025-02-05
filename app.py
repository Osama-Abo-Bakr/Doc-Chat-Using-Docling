from typing import Iterator
import time
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tempfile
import streamlit as st
from groq import Groq
from langchain_core.documents import Document as LCDocument
from langchain_core.document_loaders import BaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv



class DoclingBookLoader(BaseLoader):
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        accelerator_options = AcceleratorOptions(
            num_threads=8, device=AcceleratorDevice.AUTO
        )

        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )

    def lazy_load(self) -> Iterator[LCDocument]:
        """Yield a single `LCDocument` containing the contents of the PDF file
        at `self.file_path`, converted to Markdown format.

        The `metadata` field of the yielded document contains the following
        information:

        * `source`: The path to the PDF file.
        * `format`: Always `"book"`.
        * `process_time`: The time taken to process the PDF file into a
          `DoclingDocument`.
        * `convert_time`: The time taken to convert the `DoclingDocument` to
          Markdown format.
        """
        print(f"📚 Processing book: {self.file_path}")

        process_start = time.time()
        docling_doc = self.converter.convert(self.file_path).document
        process_time = time.time() - process_start
        print(f"✅ Book processed successfully in {process_time:.2f} seconds")

        print("🔄 Converting to markdown format...")
        convert_start = time.time()
        text = docling_doc.export_to_markdown()
        st.session_state.final_text = text
        convert_time = time.time() - convert_start
        print(f"✅ Conversion complete in {convert_time:.2f} seconds")

        metadata = {
            "source": self.file_path,
            "format": "book",
            "process_time": process_time,
            "convert_time": convert_time,
        }

        yield LCDocument(page_content=text, metadata=metadata)


def summarize_text(text):
    """
    Summarizes the given text using deepseek-r1-distill-llama-70b-specdec model.

    Parameters:
    - text (str): The text to be summarized.
    - model (str): The model to use (default is 'deepseek-r1-distill-llama-70b-specdec').
    - max_tokens (int): The maximum number of tokens for the summary.

    Returns:
    - str: The summarized text.
    """

    try:
        client = Groq()
        response = client.chat.completions.create(
            # model="deepseek-r1-distill-llama-70b",
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text in Markdown format."},
                {"role": "user", "content": f"Summarize the following text in Markdown format:\n{text}"}
            ],
            temperature=0.5
        )
        summary = response.choices[0].message.content
        return summary

    except Exception as e:
        return f"An error occurred: {e}"

def loading_vector_DB(pdf_path: str):
    """
    Loads a PDF file, processes it, and builds a vector store for book QA.

    This function performs the following operations:
    1. Initializes an embedding model using OpenAI embeddings.
    2. Loads the PDF file specified by `pdf_path` using `DoclingBookLoader`.
    3. Splits the document into chunks using `RecursiveCharacterTextSplitter`.
    4. Builds a vector store from the document chunks and embeddings using FAISS.

    Args:
        pdf_path (str): The path to the PDF file to be processed.

    Returns:
        vectorstore: The constructed vector store with embeddings for the document.
    """

    print("\n🚀 Initializing Book QA System...")
    print("🔤 Initializing embedding model...")
    embedding_start = time.time()
    embeddings = OpenAIEmbeddings()
    embedding_init_time = time.time() - embedding_start
    print(f"✅ Embedding model initialized in {embedding_init_time:.2f} seconds")

    print(f"💾 Creating vector store")    
    loader = DoclingBookLoader(pdf_path)
    documents = loader.load()

    print("📄 Splitting document into chunks...")
    split_start = time.time()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
        )
    
    splits = text_splitter.split_documents(documents)
    split_time = time.time() - split_start
    print(f"✅ Created {len(splits)} chunks in {split_time:.2f} seconds")

    print("📦 Building vector store and creating embeddings...")
    vectorstore_start = time.time()
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore_time = time.time() - vectorstore_start
    print(f"✅ Vector store built in {vectorstore_time:.2f} seconds")
    
    return vectorstore

def loading_qa_chain(vectorstore):
    """
    Initializes a QA chain using a vector store retriever and a language model.

    Args:
        vectorstore: The vector store containing document embeddings.

    Returns:
        A ConversationalRetrievalChain object configured to answer questions 
        about a book using the provided vector store and language model.
    """

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    print("✅ Vector store ready")

    print("🤖 Connecting to local language model...")
    llm = ChatOpenAI(
        model="gpt-4o"
    )

    print("⛓️ Creating QA chain...")

    template = """You are a helpful assistant answering questions about the book: {book_name}. 
    
    Use the following context to answer the question: {context}
    
    Question: {question}
    
    Answer the question accurately and concisely based on the context provided."""

    prompt = PromptTemplate(
        input_variables=["book_name", "context", "question"], template=template
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": prompt,
            "document_variable_name": "context",
        },
    )

    print(f"\n✨ System ready!")
    return qa_chain




def main():
    st.set_page_config(page_title='RAG QA system for PDF Using Docling', page_icon="🤖", layout="wide")
    st.title("RAG QA system for PDF Using Docling 🤖")
    
    load_dotenv()
    
    uploaded_file = st.sidebar.file_uploader("Upload your PDF", type=['pdf', 'png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:        

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name
            
        
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = loading_vector_DB(pdf_path=temp_path)
            st.session_state.summary = summarize_text(text=st.session_state.final_text)
        if "qa_system" not in st.session_state:
            st.session_state.qa_system = loading_qa_chain(vectorstore=st.session_state.vector_store)
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        if 'summary' in st.session_state:
            st.sidebar.subheader("Summary:")
            st.sidebar.write(st.session_state.summary)  
        else: pass
        
        question = st.chat_input("Ask a question:")
        if question:
            with st.spinner("Processing your question..."):
            
                result = st.session_state.qa_system.invoke(
                    {
                        "question": question,
                        "chat_history": st.session_state.chat_history,
                        "book_name": os.path.basename(temp_path),
                    }
                )
            
            st.session_state.chat_history.append((question, result["answer"]))
            
            for chat in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(chat[0])
                      
                with st.chat_message("assistant"):
                    st.write(chat[1])
                    
                with st.expander("References Documents"):
                        for i, doc in enumerate(result["source_documents"], 1):
                            st.write(f"Document {i}:")
                            st.write(doc.page_content)
                            st.write("---")
        
        # Button to download final_text as Markdown
        if 'final_text' in st.session_state:
            st.sidebar.download_button(
                label="Download The Summary as Markdown",
                data=st.session_state.summary,
                file_name="final_text.md",
                mime="text/markdown"
            )


if __name__ == "__main__":
    main()