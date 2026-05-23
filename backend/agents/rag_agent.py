from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from backend.config import config
from backend.rag.vector_store import VectorStoreManager
import json
from backend.utils.logger import get_logger
from backend.utils.retry import async_retry

logger = get_logger(__name__)

class KnowledgeBaseRetrievalAgent:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL
        )
        
        self.vectorstore = Chroma(
            collection_name=config.CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=config.CHROMA_PERSIST_DIR
        )
        
        self.llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model=config.MODEL_NAME,
            temperature=0.2
        )
        
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": config.MAX_RETRIEVAL_DOCS}
        )
        self.store_manager = VectorStoreManager()
    
    async def retrieve_knowledge(self, query: str, intent: str) -> List[Dict[str, Any]]:
        # Enhance query with intent context and perform intent-filtered retrieval.
        enhanced_query = f"Intent: {intent}. Customer question: {query}"
        retrieved_docs = []

        try:
            docs = self.vectorstore.similarity_search(
                enhanced_query,
                k=config.MAX_RETRIEVAL_DOCS,
                filter={"intent": intent}
            )
            if not docs:
                logger.warning("Intent-filtered retrieval returned no docs; falling back to general retrieval.")
                docs = self.vectorstore.similarity_search(
                    enhanced_query,
                    k=config.MAX_RETRIEVAL_DOCS
                )
        except Exception as e:
            logger.warning(f"Intent-filtered retrieval failed: {e}")
            docs = self.vectorstore.similarity_search(
                enhanced_query,
                k=config.MAX_RETRIEVAL_DOCS
            )

        for doc in docs:
            content = getattr(doc, "page_content", str(doc))
            metadata = getattr(doc, "metadata", {})
            relevance = metadata.get("score", 1.0)
            retrieved_docs.append({
                "content": content,
                "metadata": metadata,
                "relevance_score": relevance
            })

        logger.info(f"retrieve_knowledge returned {len(retrieved_docs)} docs for query")
        return retrieved_docs
    
    async def answer_with_context(self, query: str, intent: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        context = "\n\n".join([doc["content"] for doc in docs])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable customer support agent.
            Use the following context from the knowledge base to answer the customer's question.
            If the answer is not in the context, say so clearly.
            
            Context:
            {context}
            
            Provide:
            1. Main answer
            2. Confidence level (0-1)
            3. Sources used
            4. Missing information (if any)"""),
            ("human", "Intent: {intent}\nQuestion: {query}")
        ])
        
        chain = prompt | self.llm
        response = await async_retry(
            lambda: chain.ainvoke({
                "context": context,
                "intent": intent,
                "query": query
            }),
            retries=3,
            initial_delay=0.5,
            backoff_factor=2.0,
        )
        
        return {
            "answer": response.content,
            "retrieved_docs": docs,
            "num_docs": len(docs)
        }

    async def get_next_expected_query(self, current_intent: str, current_sentiment: str) -> Dict[str, Any]:
        return self.store_manager.get_next_expected_query(current_intent, current_sentiment)
