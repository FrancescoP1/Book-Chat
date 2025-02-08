import os
import streamlit as st
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions, PdfPipelineOptions
from docling.document_converter import PdfFormatOption
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


def list_books(folder="books"):
    return [f for f in os.listdir(folder) if f.endswith(".pdf")]

class DoclingDocumentLoader:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        accelerator_options = AcceleratorOptions(num_threads=8, device=AcceleratorDevice.AUTO)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options
        pipeline_options.generate_page_images = True
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        self.converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})

    def lazy_load(self):
        docling_doc = self.converter.convert(self.file_path).document
        text = docling_doc.export_to_markdown()
        metadata = {"source": self.file_path, "format": "book"}
        yield LCDocument(page_content=text, metadata=metadata)


def init_chat_system(pdf_path: str):
    index_path = f"{pdf_path}_faiss_index"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(index_path):
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        loader = DoclingDocumentLoader(pdf_path)
        documents = list(loader.lazy_load())
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])
        splits = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(index_path)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    llm = ChatOpenAI(model="local-model", openai_api_base="http://localhost:1234/v1", openai_api_key="not-needed", temperature=0)
    template = """You are a helpful assistant answering questions about the book: {book_name}.\n\nUse the following context to answer the question: {context}\n\nQuestion: {question}\n\nAnswer the question accurately and concisely based on the context provided."""
    prompt = PromptTemplate(input_variables=["book_name", "context", "question"], template=template)
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=True, combine_docs_chain_kwargs={"prompt": prompt, "document_variable_name": "context"})
    return qa_chain


def main():
    st.set_page_config(page_title="Book Chat")
    st.title("📚 Book Chat")
    books = list_books()
    selected_book = st.selectbox("Select a book:", books, index=None)
    uploaded_file = st.file_uploader("Or upload a PDF:", type=["pdf"])
    if uploaded_file is not None:

        file_path = os.path.join("books", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    # Initialize session state variables if they don't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "qa_system" not in st.session_state:
        st.session_state.qa_system = None
    if "last_pdf_path" not in st.session_state:
        st.session_state.last_pdf_path = None
    if "question_input" not in st.session_state:
        st.session_state.question_input = ""
    if "book_processed" not in st.session_state:
        st.session_state.book_processed = False

    # Detect if book is changed or uploaded file is new
    if selected_book or uploaded_file:
        current_pdf_path = os.path.join("books", selected_book) if selected_book else file_path
        if st.session_state.last_pdf_path != current_pdf_path:
            # Reset chat history and question input when the book/file changes
            st.session_state.chat_history = []
            st.session_state.qa_system = None
            st.session_state.question_input = ""
            st.session_state.book_processed = False
        
        st.session_state.last_pdf_path = current_pdf_path
        st.write(f"**Selected Book:** {selected_book if selected_book else file_path}")
    else:
        # Clear chat history and question input if no book is selected or uploaded
        st.session_state.chat_history = []
        st.session_state.qa_system = None
        st.session_state.question_input = ""
        st.session_state.last_pdf_path = None
        st.session_state.book_processed = False

    if st.button("Process Book"):
        if selected_book or uploaded_file:
            with st.spinner("Processing book..."):
                pdf_path = os.path.join("books", selected_book) if selected_book else file_path
                qa_system = init_chat_system(pdf_path)
                st.session_state.qa_system = qa_system
                st.session_state.chat_history = []
                st.session_state.book_processed = True
                st.success("Book processed! You can now ask questions.")
        else:
            st.error("Please select a book or upload a PDF.")
    
    # Only display chat history and question input if the book is processed
    if st.session_state.book_processed and "qa_system" in st.session_state:
        st.write("### Chat History")
        for q, a in st.session_state.chat_history:
            st.write(f"**Q:** {q}")
            st.write(f"**A:** {a}")
        
        # Use the session state directly for the input field
        question = st.text_input("Ask a question about the book:", value=st.session_state.question_input)
        
        if st.button("Get Answer") and question:
            with st.spinner("Fetching answer..."):
                result = st.session_state.qa_system.invoke({"question": question, "chat_history": st.session_state.chat_history, "book_name": selected_book if selected_book else "Uploaded Book"})
                st.session_state.chat_history.append((question, result["answer"]))
                st.write("**Answer:**", result["answer"])
                
                # Clear the question input after showing the answer
                st.session_state.question_input = ""

if __name__ == "__main__":
    main()