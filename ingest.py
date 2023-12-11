import os 
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader

class ingest:
    def __init__(self, file_path, store_directory="stores/lit_cosine"): 

        print("we re here")
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name = "BAAI/bge-large-en",
            model_kwargs = {'device':'cpu'},
            encode_kwargs = {'normalize_embeddings':False}
        )
        print("embeddings complete ")

        self.loader = PyPDFLoader(file_path)
        self.document = self.loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=350 , chunk_overlap=25)
        self.text = text_splitter.split_documents(self.document) 

        vector_store = Chroma.from_documents(
            self.text,
            self.embeddings,
            collection_metadata = {"hnsw:space": "cosine"},
            persist_directory = "stores/lit_cosine"
        )


    


    
    

    
    




