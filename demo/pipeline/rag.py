import os
import sys
import logging
logger = logging.getLogger(__name__)

log_dir = "./logs"
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



from typing import List, Optional
import qdrant_client
from hashlib import sha256
from llama_index.core.llms.llm import LLM
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core import (
    QueryBundle,
    PromptTemplate,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.vector_stores import MetadataFilters
from llama_index.core.retrievers import BaseRetriever

from custom.template import QA_TEMPLATE




class QdrantRetriever(BaseRetriever):
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embed_model: BaseEmbedding,
        similarity_top_k: int = 2,
        filters: Optional[MetadataFilters] = None 
    ) -> None:
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._similarity_top_k = similarity_top_k
        self._filters = filters

        super().__init__()

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding, similarity_top_k=self._similarity_top_k, filters = self._filters # change2: add filter
        )
        query_result = await self._vector_store.aquery(vector_store_query)

        node_with_scores = []
        for node, similarity in zip(query_result.nodes, query_result.similarities):
            node_with_scores.append(NodeWithScore(node=node, score=similarity))
        return node_with_scores

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding, similarity_top_k=self._similarity_top_k, filters = self._filters
        )
        query_result = self._vector_store.query(vector_store_query)

        node_with_scores = []
        for node, similarity in zip(query_result.nodes, query_result.similarities):
            node_with_scores.append(NodeWithScore(node=node, score=similarity))
        return node_with_scores

    def update_filters(self, new_filters: Optional[MetadataFilters]) -> None:
        """
        Update the filters used for retrieval.
        
        Args:
        - new_filters (Optional[MetadataFilters]): New filters to apply.
        """
        self._filters = new_filters
        


def hash_node(node):
    doc_identity = str(node.text)
    doc_identity = str(sha256(doc_identity.encode("utf-8", "surrogatepass")).hexdigest())
    return doc_identity


class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever, similarity_top_k = 2):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle):
        bm25_nodes = self.bm25_retriever.retrieve(QueryBundle)
        vector_nodes = self.vector_retriever.retrieve(query_bundle)
        bm25_nodes = [node for node in bm25_nodes if node.score > 0] # make sure score > 0

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes[:self.similarity_top_k]
    
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        bm25_nodes =  self.bm25_retriever.retrieve(query_bundle)
        vector_nodes = await self.vector_retriever.aretrieve(query_bundle)
        # bm25_nodes = await self.vector_retriever.aretrieve(query_bundle)
        bm25_nodes = [node for node in bm25_nodes if node.score > 0] # make sure score > 0

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes[:self.similarity_top_k]


class HybridRetrieverRRF(BaseRetriever):
    """Retriever that ensembles the multiple retrievers.

    It uses a rank fusion.

    Args:
        retrievers: A list of retrievers to ensemble.
        weights: A list of weights corresponding to the retrievers. Defaults to equal
            weighting for all retrievers.
        k: A constant added to the rank, controlling the balance between the importance
            of high-ranked items and the consideration given to lower-ranked items.
            Default is 60.
        id_key: The key in the document's metadata used to determine unique documents.
            If not specified, page_content is used.
    """
    def __init__(self, vector_retriever, bm25_retriever, similarity_top_k= 2, k=60, bm25_weight=0.5 ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.k = k  # Parameter for RRF
        self.similarity_top_k = similarity_top_k

        assert 0 <= bm25_weight <= 1, "bm25_weight must be between 0 and 1"
        self.bm25_weight = bm25_weight
        self.vector_weight = 1 - bm25_weight
        super().__init__()

         


    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)
        vector_nodes = self.vector_retriever.retrieve(query_bundle)
        bm25_nodes = [node for node in bm25_nodes if node.score > 0] # make sure score > 0

        # Assign ranks to the nodes
        # bm25_ranks = {node.node.node_id: rank for rank, node in enumerate(bm25_nodes, 1)}
        # vector_ranks = {node.node.node_id: rank for rank, node in enumerate(vector_nodes, 1)}
        bm25_ranks = {hash_node(node.node): rank for rank, node in enumerate(bm25_nodes, 1)}
        vector_ranks = {hash_node(node.node): rank for rank, node in enumerate(vector_nodes, 1)}


        # Combine the scores using RRF
        all_nodes = {}
        for node in bm25_nodes:
            # node_id = node.node.node_id
            node_id = hash_node(node.node)
            rank = bm25_ranks[node_id]
            score = 1 * self.bm25_weight / (self.k + rank)
            if node_id in all_nodes:
                all_nodes[node_id].score += score
            else:
                all_nodes[node_id] = NodeWithScore(node=node.node, score=score)
        
        for node in vector_nodes:
            # node_id = node.node.node_id
            node_id = hash_node(node.node)
            rank = vector_ranks[node_id]
            score = 1 * self.vector_weight / (self.k + rank)
            if node_id in all_nodes:
                all_nodes[node_id].score += score
            else:
                all_nodes[node_id] = NodeWithScore(node=node.node, score=score)

        # Sort the combined nodes by their final score
        combined_nodes = sorted(all_nodes.values(), key=lambda x: x.score, reverse=True)
        # logger.info(f"combined_nodes: {str(combined_nodes)}")

        return combined_nodes[:self.similarity_top_k]
    
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)
        vector_nodes = await self.vector_retriever.aretrieve(query_bundle)
        bm25_nodes = [node for node in bm25_nodes if node.score > 0] # make sure score > 0
        vector_scores = str([node.score for node in vector_nodes ])
        bm25_scores = str([node.score for node in bm25_nodes ])

        logger.info(f"semantic search nodes: {len(vector_nodes)} | scores: {vector_scores}")
        logger.info(f"sparse search nodes: {len(bm25_nodes)} | scores: {bm25_scores}")
        node_ids1 = [hash_node(node.node) for node in bm25_nodes]
        node_ids2 = [hash_node(node.node) for node in vector_nodes]
        # Convert lists to sets
        set1 = set(node_ids1)
        set2 = set(node_ids2)

        # Find intersection of sets
        intersection = set1.intersection(set2)

        # Calculate number of elements in intersection
        intersection_count = len(intersection)
        logger.info(f"intersection_count: {intersection_count}")


        # Assign ranks to the nodes
        # bm25_ranks = {node.node.node_id: rank for rank, node in enumerate(bm25_nodes, 1)}
        # vector_ranks = {node.node.node_id: rank for rank, node in enumerate(vector_nodes, 1)}
        bm25_ranks = {hash_node(node.node): rank for rank, node in enumerate(bm25_nodes, 1)}
        vector_ranks = {hash_node(node.node): rank for rank, node in enumerate(vector_nodes, 1)}


        # Combine the scores using RRF
        all_nodes = {}
        for node in bm25_nodes:
            # node_id = node.node.node_id
            node_id = hash_node(node.node)

            rank = bm25_ranks[node_id]
            score = 1 * self.bm25_weight / (self.k + rank)
            if node_id in all_nodes:
                all_nodes[node_id].score += score
            else:
                all_nodes[node_id] = NodeWithScore(node=node.node, score=score)
        
        for node in vector_nodes:
            # node_id = node.node.node_id
            node_id = hash_node(node.node)

            rank = vector_ranks[node_id]
            score = 1 * self.vector_weight  / (self.k + rank)
            if node_id in all_nodes:
                all_nodes[node_id].score += score
                logger.info("bm25 nodes exists, merge nodes score.")
            else:
                all_nodes[node_id] = NodeWithScore(node=node.node, score=score)

        # Sort the combined nodes by their final score
        combined_nodes = sorted(all_nodes.values(), key=lambda x: x.score, reverse=True)
        # logger.info(f"combined_nodes: {str(combined_nodes)}")
        
        return combined_nodes[:self.similarity_top_k]


def reciprocal_rank_fusion(nodes_lists, k=60, weights=None, similarity_top_k = 2):
    """
    Combine rankings from multiple ranking systems using Reciprocal Rank Fusion (RRF).
    
    Args:
        nodes_lists (list of list of Node): A list of lists where each sublist contains nodes from a ranking system.
        k (int, optional): The RRF parameter. Defaults to 60.
        weights (list of int, optional): The weights for each ranking system. Defaults to None.
        
    Returns:
        list of NodeWithScore: A sorted list of nodes with combined scores.
    """
    if weights is None:
        weights = [1] * len(nodes_lists)

    all_nodes = {}

    for idx, nodes in enumerate(nodes_lists):
        node_ranks = {hash_node(node.node): rank for rank, node in enumerate(nodes, 1)}
        weight = weights[idx]

        for node in nodes:
            node_id = hash_node(node.node)
            rank = node_ranks[node_id]
            score = 1 * weight  / (k + rank)

            if node_id in all_nodes:
                all_nodes[node_id].score += score
            else:
                all_nodes[node_id] = NodeWithScore(node=node.node, score=score)

    combined_nodes = sorted(all_nodes.values(), key=lambda x: x.score, reverse=True)

    return combined_nodes[:similarity_top_k]


async def generation_with_knowledge_retrieval(
    query_str: str,
    retriever: BaseRetriever,
    # retriever: QdrantRetriever,
    llm: LLM,
    qa_template: str = QA_TEMPLATE,
    reranker: BaseNodePostprocessor | None = None,
    debug: bool = False,
    progress=None,
    query_doc = None, # query的来源，如果非None，就按照来源过滤，仅用那个类目下的文档。
    multi_quries = [] # 多个query，各自检索文档后合并
) -> CompletionResponse:
    if query_doc is not None:
        # filter
        from llama_index.core.vector_stores import MetadataFilters, FilterCondition
        metadata_dicts = [{"key":"document", "value":query_doc}]
        metadata_filter = MetadataFilters.from_dicts(metadata_dicts, condition=FilterCondition.OR)
        retriever.update_filters(metadata_filter)
    
    if not multi_quries:
        query_bundle = QueryBundle(query_str=query_str)
        node_with_scores = await retriever.aretrieve(query_bundle)
    else:
        # multi query retrieve
        logger.info("execute multi query retrieve.")
        # node_with_scores = []
        # seen = set()
        nodes_list = []
        for single_query in multi_quries:
            query_bundle = QueryBundle(query_str=single_query)
            # nodes = await retriever.aretrieve(query_bundle) # node_with_score
            logger.info(f"single_query: {single_query}")
            nodes = await retriever.aretrieve(query_bundle) # node_with_score

            # # dedup 
            # for node in nodes:
            #     if node.node.node_id not in seen:
            #         seen.add(node.node.node_id)
            #         node_with_scores.append(node)

            nodes_list.append(nodes)

        # node_with_scores = reciprocal_rank_fusion(nodes_list, weights=[1]*len(nodes_list), similarity_top_k=retriever.similarity_top_k)
        node_with_scores = reciprocal_rank_fusion(nodes_list, weights=[3,2,1], similarity_top_k=retriever.similarity_top_k)
        

        logger.info(f"multi query retrieve nodes: {len(node_with_scores)}")
        # recall_topk = retriever.similarity_top_k
        # node_with_scores = node_with_scores[:recall_topk]
        # logger.info(f"multi query retrieve nodes(topk): {len(node_with_scores)}")
        retrieve_scores = [node.score for node in node_with_scores ]
        mean_score = sum(retrieve_scores) / len(retrieve_scores)
        logger.info(f"retrieve scores: {str(retrieve_scores)}, mean scores: {mean_score}")

    if debug:
        print(f"retrieved:\n{node_with_scores}\n------")
    if reranker:
        node_with_scores = reranker.postprocess_nodes(node_with_scores, query_bundle)
        if debug:
            print(f"reranked:\n{node_with_scores}\n------")
        # 打印分数
        rerank_scores = [node.score for node in node_with_scores ]
        mean_score = sum(rerank_scores) / len(rerank_scores)
        logger.info(f"reranked scores: {str(rerank_scores)}, mean scores: {mean_score}")

    context_str = "\n\n".join(
        [f"{node.metadata['document_title']}: {node.text}" for node in node_with_scores]
    )
    # print(context_str)
    docs = [node.metadata.get("document", "None") for node in node_with_scores]
    print("node_with_scores docs source: ", set(docs))


    fmt_qa_prompt = PromptTemplate(qa_template).format(
        context_str=context_str, query_str=query_str
    )
    ret = await llm.acomplete(fmt_qa_prompt)

    dev = True
    if dev:
        logger.info(f"Query: {query_str}")
        logger.info(f"Answer: \n{ret}")
        # logger.info(f"Prompt: \n{fmt_qa_prompt}")
        CODE_ROOT = os.getenv("CODE_ROOT")
        get_path = lambda node: node.metadata["file_path"].replace(CODE_ROOT+"/data/", "")
        context_str_wpath = "\n\n".join(
            [f"File path: {get_path(node)}\n Document title: {node.metadata['document_title']}: \n Text: {node.text}" for node in node_with_scores]
        )
        # logger.info(f"Detail: \n{context_str_wpath}")
        

    
    if progress:
        progress.update(1)
    return ret

