import os
import sys
import logging
import argparse

logger = logging.getLogger(__name__)

log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(log_dir,"log.txt")),
    ],
}
logging.basicConfig(**LOG_CONFIG)
logger = logging.getLogger(__name__)

import re
import jieba
import asyncio

from dotenv import dotenv_values
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from qdrant_client import models
from tqdm.asyncio import tqdm
from llama_index.core import StorageContext
from qdrant_client.models import Distance, VectorParams
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.node_parser import SentenceSplitter

from pipeline.ingestion import build_pipeline, build_vector_store, read_data
from pipeline.qa import read_jsonl, save_answers
from pipeline.rag import QdrantRetriever, generation_with_knowledge_retrieval
from llama_index.postprocessor.cohere_rerank import CohereRerank

from pipeline.rag import HybridRetriever, HybridRetrieverRRF


CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["CODE_ROOT"] = CODE_ROOT



def chinese_tokenizer(text):
    # 去除不需要的符号，这里我们只保留汉字、数字、字母和常用符号
    cleaned_text = re.sub(r'[^\u4e00-\u9fa5\w]', ' ', text)
    
    # 使用jieba进行分词
    tokens = jieba.cut(cleaned_text)
    tokens = [tok.strip() for tok in list(tokens) if tok.strip()!=""]
    return tokens



async def main(args):
    # recall_topk = 6
    # topk = 4
    recall_topk = 8
    topk = 5
    use_mquery = True
    use_HyDE = True

    config = dotenv_values(".env")
    # 初始化 LLM 嵌入模型 和 Reranker
    llm = Ollama(
        model="qwen", base_url=config["OLLAMA_URL"], temperature=0, request_timeout=120
    )
    # change1: move to cuda
    rerank_path = "/home/jhx/Projects/pretrained_models/bge-reranker-v2-m3/"
    embed_path = "/home/jhx/Projects/pretrained_models/bge-m3/"
    embed_args = {
        'model_name': embed_path, 
        'max_length': 1024,  
        'embed_batch_size': 32,
        'device': 'cuda'
        }

    embeding = HuggingFaceEmbedding(**embed_args)
    Settings.embed_model = embeding
    Settings.llm = llm

    from llama_index.core.postprocessor import SentenceTransformerRerank
    # We choose a model with relatively high speed and decent accuracy.
    reranker = SentenceTransformerRerank(
        model=rerank_path, top_n=topk, device="cuda"
    )
    # cohere_rerank = CohereRerank(api_key=config["COHERE_API_KEY"], top_n=topk) # 从重排器返回前 2 个节点


    # 初始化 数据ingestion pipeline 和 vector store
    print("build_vector_store")
    client, vector_store = await build_vector_store(config, reindex=False)
    print("get_collection")
    collection_info = await client.get_collection(
        config["COLLECTION_NAME"] or "aiops24"
    )

    print("collection_info.points_count",collection_info.points_count)
    data = read_data("data")


    if collection_info.points_count == 0:
    # if collection_info.points_count < 10**9:
        pipeline = build_pipeline(llm, embeding, vector_store=vector_store)
        # 暂时停止实时索引
        print("update_collection1")
        await client.update_collection(
            collection_name=config["COLLECTION_NAME"] or "aiops24",
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )
        # await pipeline.arun(documents=data, show_progress=True, num_workers=1) # >1 卡住
        nodes = await pipeline.arun(documents=data, show_progress=True, num_workers=1)

        # 恢复实时索引
        print("update_collection2")
        await client.update_collection(
            collection_name=config["COLLECTION_NAME"] or "aiops24",
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
        )
        print("update_collection over")
        print(len(data))
    else:
        print("build pipeline else")
        pipeline = build_pipeline(llm, embed_model=None, vector_store=None)
        # # nodes = pipeline.run(documents=data, show_progress=True, num_workers=1)
        print("get nodes arun")
        import time
        s1 = time.time()
        nodes = await pipeline.arun(documents=data, show_progress=False, num_workers=1)
        s2 = time.time()
        print(f"arun over, took {s2-s1} secs...")


    # retireve the top 10 most similar nodes using bm25
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=recall_topk, tokenizer=chinese_tokenizer)
    # retireve the top 10 most similar nodes using embeddings
    retriever = QdrantRetriever(vector_store, embeding, similarity_top_k=recall_topk)


    hybrid_retriever = HybridRetrieverRRF(retriever, bm25_retriever, bm25_weight = 0.4, similarity_top_k=recall_topk) # 0.2 4 6 8 


    queries = read_jsonl("question.jsonl")
    aug_queries = read_jsonl("question_rewrite.jsonl")


    # 生成答案
    logger.info("Start generating answers...")

    results = []

    # multi query retrieve
    for qid in tqdm(range(len(queries)), total=len(queries)):
        query = queries[qid]["query"]
        multi_quries = [query, aug_queries[qid]["query"]] if use_mquery else []
        # Hypothetical Document Embeddings（HyDE）
        if use_HyDE:
            qa_file = "submit_result_fusion_rrf_b6v6r4_bw4_lc_splitter.jsonl"
            qa_list = read_jsonl(qa_file)
            answer = qa_list[qid]["answer"]
            multi_quries.append(answer)

        result = await generation_with_knowledge_retrieval(
                    query, hybrid_retriever, llm, reranker=reranker,
                    multi_quries = multi_quries,
                    # query_doc = query["document"]
                )
        results.append(result)


    # 处理结果
    save_answers(queries, results, "submit_result.jsonl")



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recall_topk", type=int, default=8)
    parser.add_argument("--rerank_topk", type=int, default=4)
    parser.add_argument("--bm25_weight", type=float, default=0.5)
    

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = get_args()
    asyncio.run(main(args))
