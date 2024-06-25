import os
from typing import List

from llama_index.core import SimpleDirectoryReader
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.extractors import SummaryExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms.llm import LLM
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, MetadataMode
from llama_index.core.readers.file.base import default_file_metadata_func
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from custom.template import SUMMARY_EXTRACT_TEMPLATE
from custom.transformation import CustomFilePathExtractor, CustomTitleExtractor

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter



# 定义一个函数，用于设置文件元数据
def filename_hook(filename):
    meta_data = default_file_metadata_func(filename)
    CODE_ROOT = os.getenv("CODE_ROOT")
    document = filename.replace(CODE_ROOT+"/data/", "").strip().split("/")[0]
    meta_data["document"] = document

    return meta_data

def read_data(path: str = "data") -> list[Document]:
    reader = SimpleDirectoryReader(
        input_dir=path,
        recursive=True,
        required_exts=[
            ".txt",
        ],
        file_metadata=filename_hook
    )
    return reader.load_data()


text_splitter = RecursiveCharacterTextSplitter(chunk_size= 1024, chunk_overlap=50)

class LCChineseSplitter(SentenceSplitter):
    # https://github.com/chatchat-space/Langchain-Chatchat/issues/996
    # def __init__(self, sentence_size: int = 64, **kwargs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.sentence_size = sentence_size
        # self.text_splitter = RecursiveCharacterTextSplitter(chunk_size= self.chunk_size, chunk_overlap=self.chunk_overlap)
        print("init langchain splitter")

    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """
        _Split incoming text and return chunks with overlap size.

        Has a preference for complete sentences, phrases, and minimal overlap.
        """
        global text_splitter
        if text == "":
            return [text]
        # chunks = self.text_splitter.split_text(text)
        chunks = text_splitter.split_text(text)

        return chunks



def build_pipeline(
    llm: LLM,
    embed_model: BaseEmbedding,
    template: str = None,
    vector_store: BasePydanticVectorStore = None,
    chunk_size = 1024,
    chunk_overlap = 50,
) -> IngestionPipeline:
    transformation = [
        # SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        LCChineseSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        CustomTitleExtractor(metadata_mode=MetadataMode.EMBED),
        CustomFilePathExtractor(last_path_length=4, metadata_mode=MetadataMode.EMBED),
        # SummaryExtractor(
        #     llm=llm,
        #     metadata_mode=MetadataMode.EMBED,
        #     prompt_template=template or SUMMARY_EXTRACT_TEMPLATE,
        # ),
        # embed_model,
    ]
    if embed_model is not None:
        transformation.append(embed_model)

    return IngestionPipeline(transformations=transformation, vector_store=vector_store)


async def build_vector_store(
    config: dict, reindex: bool = False
) -> tuple[AsyncQdrantClient, QdrantVectorStore]:
    api_key=config.get("QDRANT_KEY", None)
    
    print("api key:", api_key)
    client = AsyncQdrantClient(
        # location=":memory:"
        url=config["QDRANT_URL"],
        api_key=config.get("QDRANT_KEY", None),

    )
    if reindex:
        try:
            await client.delete_collection(config["COLLECTION_NAME"] or "aiops24")
        except UnexpectedResponse as e:
            print(f"Collection not found: {e}")

    try:
        await client.create_collection(
            collection_name=config["COLLECTION_NAME"] or "aiops24",
            vectors_config=models.VectorParams(
                size=config["VECTOR_SIZE"] or 1024, distance=models.Distance.DOT
            ),
        )
    except UnexpectedResponse:
        print("Collection already exists")
    return client, QdrantVectorStore(
        aclient=client,
        collection_name=config["COLLECTION_NAME"] or "aiops24",
        parallel=4,
        batch_size=32,
    )
