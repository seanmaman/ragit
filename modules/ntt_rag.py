import datetime
import logging
import mimetypes
import platform

import os
import sys
import chromadb
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader,VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.readers.base import BaseReader
from llama_index.core import Document

from modules.ntt_utils import NTTUtils

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
log = logging.getLogger(__name__)

"""""
LLAMAINDEX SUPPORTED FILES TYPES
================================
.csv - comma-separated values
.docx - Microsoft Word
.epub - EPUB ebook format
.hwp - Hangul Word Processor
.ipynb - Jupyter Notebook
.jpeg, .jpg - JPEG image
.mbox - MBOX email archive
.md - Markdown
.mp3, .mp4 - audio and video
.pdf - Portable Document Format
.png - Portable Network Graphics
.ppt, .pptm, .pptx - Microsoft PowerPoint
"""""
class NTTRAG:
    
    def __init__(self, settings):

        self.settings = settings
        self.db = chromadb.PersistentClient(path=self.settings["db_path"])
        self.collection = self.db.get_or_create_collection(self.settings["collection_name"])
        self.vs = ChromaVectorStore(chroma_collection=self.collection)
        self.sc = StorageContext.from_defaults(vector_store=self.vs)
        self.embed_model = HuggingFaceEmbedding(model_name=self.settings["embed_model"])
        self.llm = Ollama(model=self.settings["llm_model"], 
                          base_url= self.settings["ollama_host"],
                          request_timeout=self.settings["request_timeout"])
        self.index = VectorStoreIndex.from_vector_store(vector_store=self.vs, 
                                                        embed_model=self.embed_model)
 
    def query(self,query_str):
        qe = self.index.as_query_engine(llm=self.llm)
        try:
            prompt = f"{query_str}"
            response = qe.query(prompt)
            return response.response
        except Exception as e:
            log.error(f"Query failed: {e}")

    def query_with_sources(self,query_str,metadata_filters=None):
        #perform a query and return both the response and source references
        qe = self.index.as_query_engine(
            llm=self.llm,
            response_mode="tree_summarize",  # Provides structured response with sources
            verbose=True,
            streaming=True,
            node_postprocessors=[],
            filters=metadata_filters,
            #similarity_top_k=5,
        )

        prompt = f"{query_str}"
        response = qe.query(prompt)
            
        # Extract source information
        """""
        metadata example                
        'page_label' = '2'
        'file_name' = '07.01.25 Jpost Daily.pdf'
        'file_path' = 'E:\\Projects\\nttragger\\documents\\07.01.25 Jpost Daily.pdf'
        'file_type' = 'application/pdf'
        'file_size' = 40007952
        'creation_date' = '2025-01-06'
        'last_modified_date' = '2025-01-06'
        """
        sources = []
        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                source_info = {
                    "text": node.node.text,
                    "score": round(node.score, 3) if hasattr(node, "score") else None,
                    "metadata": node.node.metadata if hasattr(node.node, "metadata") else {},
                }
                sources.append(source_info)
        
        return {
            "response": str(response),
            "sources": sources
        }  

    @staticmethod
    def get_file_metadata(file_path):
        
        file_stats = os.stat(file_path)
        
        metadata = {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "file_type": mimetypes.guess_type(file_path)[0] or "application/octet-stream",
        "file_size": file_stats.st_size,
        "creation_date": datetime.datetime.fromtimestamp(file_stats.st_ctime).strftime("%Y-%m-%d"),
        "last_modified_date": datetime.datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d'),
        }

        # Extract folder path and split into individual folders
        folder_path = os.path.dirname(file_path)
        folders = folder_path.split(os.sep)  # Use os.sep for platform compatibility
        # Filter out empty strings that can occur from leading/trailing separators
        folders = [folder for folder in folders if folder]
        # Remove drive letter on Windows
        if platform.system() == "Windows" and len(folders) > 0 and folders[0].endswith(":"):
            folders = folders[1:]  # Slice the list to remove the first element
       
        #metadata["tags"] = folders #llamaindex doest allow arrays in metadata
        metadata["client"] = folders[-1]
        
        return metadata
    
    def add_file(self,file_path,reader=None):
        status = self.find_file_documents(file_path)
        if status == "exist":
            return #skip db insertion

        if reader==None:
            reader = SimpleDirectoryReader(input_files=[file_path] ,
                                        file_extractor={
                                            ".mp3": AudioVideoFileReader(),
                                            ".mp4": AudioVideoFileReader(),
                                            ".ts": AudioVideoFileReader()
                                            },
                                            file_metadata=NTTRAG.get_file_metadata
                                            )

        if status == "not exist":
            log.debug(f"extracting text from {file_path}...")
            documents = reader.load_data() #remember for one file many documents per chunks
            log.debug(f"extracting text from {file_path} done.")
            self.insert_documents(documents)
        if status == "different":
            #file change delete and add
            log.debug(f"extracting text from {file_path}...")
            documents = reader.load_data(file_paths=[file_path])
            log.debug(f"extracting text from {file_path} done.")
            self.update_documents(documents)

    def rebuild(self,input_dir):
        log.debug(f"rebuild from {input_dir} ...")
        
        reader = SimpleDirectoryReader(input_dir=input_dir ,
                                       recursive=True,
                                       file_extractor={
                                           ".mp3": AudioVideoFileReader(),
                                           ".mp4": AudioVideoFileReader(),
                                           ".ts": AudioVideoFileReader()
                                        },
                                        #file_metadata=get_meta)
                                        )
        
        file_paths = reader.input_files
        
        for file_path in file_paths:
            self.add_file(str(file_path),reader)
        
        log.debug(f"rebuild from {input_dir} done.")

    def insert_documents(self, documents):
        try:
            file_path = documents[0].metadata.get("file_path")
            log.debug(f"insert_documents {file_path} ...")
            VectorStoreIndex.from_documents(documents, storage_context=self.sc, embed_model=self.embed_model)
            log.debug(f"insert_documents {file_path} done.")
        except Exception as e:
            log.error(f"insert_documents: {file_path} {e}")
   
    def update_documents(self, documents):
        try:
            file_path = documents[0].metadata.get("file_path")
            log.debug(f"update_documents {file_path} ...")
            self.delete_documents(file_path)
            self.insert_documents(documents)
            log.debug(f"update_documents {file_path} done.")
        except Exception as e:
            log.error(f"update_documents: {file_path} {e}")

    def delete_documents(self, file_path):
        #remove all existing entries for a document by its file path from database
        try:
            log.debug(f"delete_documents {file_path} ...")
            results = self.collection.get(where={"file_path": file_path})
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                log.warning(f'removed {len(results["ids"])} existing entries for: {file_path}')
            log.debug(f"delete_documents {file_path} done.")
        except Exception as e:
            log.error(f"delete_documents: {file_path} {e}")

    def find_file_documents(self, file_path):
        try:
            results = self.collection.get(where={"file_path": file_path})   
            if not results["ids"]:
                return "not exist"
            current_size = os.path.getsize(file_path)
            stored_size = results["metadatas"][0].get("file_size", 0)
            return "exist" if current_size == stored_size else "different"
        except Exception as e:
            log.error(f"find_file_documents: {file_path} {e}")
            return False

    def print_response_with_sources(self, prompt, result):
        response = result["response"]
        sources = result["sources"]
        
        print("\nPrompt:")
        print("-" * 80)
        print(prompt)
        print("-" * 80)

        # Print the main response
        print("\nMain Response:")
        print("-" * 80)
        print(response)
        print("-" * 80)
        
        # Print sources
        print("\nSources:")
        print("=" * 80)
        for idx, source in enumerate(sources, 1):
            print(f"\nSource {idx}:")
            print("-" * 40)
            
            # Print metadata if available
            metadata = source.get("metadata", {})
            if metadata:
                print("Metadata:")
                for key, value in metadata.items():
                    print(f"  {key}: {value}")
            
            # Print relevance score if available
            if source.get("score"):
                print(f'Relevance Score: {source["score"]}')
            
            # Print source text
            print("\nExtracted Text:")
            print(source.get("text", "No text available"))
            print("-" * 80)

class AudioVideoFileReader(BaseReader):
    def load_data(self, file, extra_info=None):
        result = NTTUtils.transcribe(str(file))
        #extra_info is already fill with metadata fill free to extend        
        return [Document(text=result["text"] or "", extra_info=extra_info or {})]
    
class PdfFileReader(BaseReader):
    def load_data(self, file, extra_info=None):
        result = ""
        #extra_info is already fill with metadata fill free to extend        
        return [Document(text=result["text"] or "", extra_info=extra_info or {})]


