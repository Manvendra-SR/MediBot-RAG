from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from src.helper import *

extracted_data=load_pdf_file(data='Data/')

text_chunks=text_split(extracted_data)

embeddings = download_hugging_face_embeddings()

index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
vector_store.add_documents(text_chunks)

vector_store.save_local("faiss_index")