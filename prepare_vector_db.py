from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

# Khai bao bien
pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss"

#ham tao vecto db tu file pdf
def create_db_from_files():
    # khai bao loader de load toan bo thu muc data(glop: doc nhung file gi, loader_cls:loader de doc file)
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls=PyPDFLoader)

    #load tat ca file
    documents = loader.load()

    #chia van ban(chunk_size: kich co van ban, chunk_overlap: doan lap do voi doan truoc Eg: doan 1 lay tu 1-499 thi doan 2 lay tu 449-948)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=499, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    #embedding: nhung tu, bieu dien duoi dang vecto so hoc, nghia la ma hoa tu theo kieu nhung tu nghia cang gan nhau thi duoc xep cang gan nhau
    models = "models/all-MiniLM-L6-v2-f16.gguf"
    embedding_model = GPT4AllEmbeddings(model_file=models)
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)

    return db

create_db_from_files()