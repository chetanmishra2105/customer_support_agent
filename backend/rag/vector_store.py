from pathlib import Path
import requests
from collections import Counter, defaultdict
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from backend.config import config
import pandas as pd
from typing import List, Dict, Any
from backend.utils.logger import get_logger

logger = get_logger(__name__)

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw_data"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

class VectorStoreManager:
    def __init__(self):
        # Lazy init: avoid downloading embedding models on import/startup.
        # Embeddings and vectorstore will be created only when needed (ingest=True).
        self.embeddings = None
        self.vectorstore = None
        self._raw_df = None
        self._conversation_chain_map = None
        self._fallback_chain_map = None

    def _init_embeddings_and_vectorstore(self):
        if self.embeddings is None:
            # logger.info("Initializing HuggingFaceEmbeddings...")
            self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        if self.vectorstore is None:
            # logger.info("Initializing Chroma vectorstore...")
            self.vectorstore = Chroma(
                collection_name=config.CHROMA_COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=config.CHROMA_PERSIST_DIR
            )

    def _normalize_text(self, text: Any) -> str:
        return str(text or "").strip()

    def _infer_sentiment_from_text(self, text: str) -> str:
        lower = text.lower()
        if any(word in lower for word in ["angry", "furious", "irate", "outraged", "mad", "upset"]):
            return "angry"
        if any(word in lower for word in ["frustrated", "annoyed", "disappointed", "fed up", "let down"]):
            return "frustrated"
        if any(word in lower for word in ["anxious", "worried", "nervous", "concerned", "panic"]):
            return "anxious"
        if any(word in lower for word in ["confused", "unclear", "not sure", "don't understand", "lost"]):
            return "confused"
        if any(word in lower for word in ["satisfied", "happy", "pleased", "thank you", "thanks", "resolved"]):
            return "positive"
        return "neutral"

    def _is_successful_response(self, row: Dict[str, Any]) -> bool:
        flags = self._normalize_text(row.get("flags", "")).lower()
        response = self._normalize_text(row.get("response", "")).lower()
        if any(term in flags for term in ["resolved", "solved", "success", "satisfied", "positive"]):
            return True
        if any(term in response for term in ["resolved", "issue fixed", "glad", "happy", "satisfied", "thank you", "thanks"]):
            return True
        return False

    def _has_escalation_flag(self, row: Dict[str, Any]) -> bool:
        flags = self._normalize_text(row.get("flags", "")).lower()
        return any(term in flags for term in ["escalated", "escalation", "human escalation", "agent", "urgent"])

    def _build_document_metadata(self, row: Dict[str, Any], source: str) -> Dict[str, Any]:
        sentiment = self._infer_sentiment_from_text(
            f"{row.get('instruction', '')} {row.get('response', '')}"
        )
        return {
            "category": self._normalize_text(row.get("category", "")),
            "intent": self._normalize_text(row.get("intent", "")).lower(),
            "source": source,
            "flags": self._normalize_text(row.get("flags", "")),
            "sentiment": sentiment,
            "successful_response": str(self._is_successful_response(row)),
            "escalation_flag": str(self._has_escalation_flag(row))
        }

    def _load_raw_df(self) -> pd.DataFrame:
        if self._raw_df is not None:
            return self._raw_df

        csv_path = RAW_DATA_DIR / "huggingface_raw_data.csv"
        if not csv_path.exists():
            return pd.DataFrame()

        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except Exception as e:
            logger.exception(f"Failed to load raw CSV for analysis: {e}")
            return pd.DataFrame()

        df = df.fillna("")
        df["intent"] = df["intent"].astype(str)
        df["category"] = df["category"].astype(str)
        df["instruction"] = df["instruction"].astype(str)
        df["response"] = df["response"].astype(str)
        df["flags"] = df["flags"].astype(str)
        df["sentiment"] = df.apply(
            lambda row: self._infer_sentiment_from_text(f"{row['instruction']} {row['response']}"),
            axis=1
        )
        df["successful_response"] = df.apply(lambda row: self._is_successful_response(row), axis=1)
        df["escalation_flag"] = df.apply(lambda row: self._has_escalation_flag(row), axis=1)

        self._raw_df = df
        return df

    def _build_conversation_chain_map(self):
        if self._conversation_chain_map is not None and self._fallback_chain_map is not None:
            return

        df = self._load_raw_df()
        self._conversation_chain_map = {}
        self._fallback_chain_map = {}

        if df.empty:
            return

        for current_index in range(len(df) - 1):
            current_intent = df.iloc[current_index]["intent"].strip().lower() or "unknown"
            current_sentiment = df.iloc[current_index]["sentiment"]
            next_intent = df.iloc[current_index + 1]["intent"].strip().lower() or "unknown"

            sentiment_key = (current_intent, current_sentiment)
            self._conversation_chain_map.setdefault(sentiment_key, Counter())[next_intent] += 1
            self._fallback_chain_map.setdefault(current_intent, Counter())[next_intent] += 1

    def _extract_fields_from_content(self, content: str) -> Dict[str, str]:
        fields = {
            "question": "",
            "category": "",
            "intent": "",
            "response": ""
        }
        for line in content.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            normalized = key.strip().lower()
            if normalized == "customer question":
                fields["question"] = value.strip()
            elif normalized == "category":
                fields["category"] = value.strip()
            elif normalized == "intent":
                fields["intent"] = value.strip()
            elif normalized == "support response":
                fields["response"] = value.strip()
        return fields

    def get_examples_by_intent(self, intent: str, limit: int = 5, include_response: bool = False) -> List[Dict[str, Any]]:
        intent_key = self._normalize_text(intent).lower()
        self._init_embeddings_and_vectorstore()

        examples = []
        try:
            documents = self.vectorstore.similarity_search(
                f"Intent examples for {intent_key}",
                k=limit,
                filter={"intent": intent_key}
            )
        except Exception as e:
            logger.warning(f"Intent-filtered retrieval failed: {e}")
            documents = []

        if documents:
            for document in documents:
                fields = self._extract_fields_from_content(document.page_content)
                example = {
                    "question": fields["question"],
                    "category": fields["category"],
                    "intent": fields["intent"],
                    "flags": document.metadata.get("flags", ""),
                }
                if include_response:
                    example["response"] = fields["response"]
                examples.append(example)

        if not examples:
            df = self._load_raw_df()
            filtered = df[df["intent"].str.lower() == intent_key].head(limit)
            for _, row in filtered.iterrows():
                example = {
                    "question": row["instruction"],
                    "category": row["category"],
                    "intent": row["intent"],
                    "flags": row["flags"],
                }
                if include_response:
                    example["response"] = row["response"]
                examples.append(example)

        return examples

    def get_responses_by_sentiment(self, sentiment: str = "neutral", limit: int = 10) -> List[Dict[str, Any]]:
        sentiment_key = self._normalize_text(sentiment).lower()
        self._init_embeddings_and_vectorstore()
        responses = []

        try:
            documents = self.vectorstore.similarity_search(
                f"Successful {sentiment_key} customer support responses",
                k=limit,
                filter={"sentiment": sentiment_key, "successful_response": "True"}
            )
        except Exception as e:
            logger.warning(f"Sentiment-filtered retrieval failed: {e}")
            documents = []

        for document in documents:
            fields = self._extract_fields_from_content(document.page_content)
            responses.append({
                "query": fields["question"],
                "intent": fields["intent"],
                "response": fields["response"],
                "flags": document.metadata.get("flags", ""),
                "sentiment": document.metadata.get("sentiment", "neutral")
            })

        if not responses:
            df = self._load_raw_df()
            filtered = df[
                (df["sentiment"] == sentiment_key) &
                (df["successful_response"] == True)
            ].head(limit)
            for _, row in filtered.iterrows():
                responses.append({
                    "query": row["instruction"],
                    "intent": row["intent"],
                    "response": row["response"],
                    "flags": row["flags"],
                    "sentiment": row["sentiment"]
                })

        return responses

    def get_next_expected_query(self, current_intent: str, current_sentiment: str = "neutral", limit: int = 3) -> Dict[str, Any]:
        self._build_conversation_chain_map()
        intent_key = self._normalize_text(current_intent).lower()
        sentiment_key = self._normalize_text(current_sentiment).lower()

        suggestions = []
        counter = self._conversation_chain_map.get((intent_key, sentiment_key))
        if not counter:
            counter = self._fallback_chain_map.get(intent_key, Counter())

        if not counter:
            return {
                "predictions": [],
                "confidence": 0.0,
                "message": "No follow-up conversation chain found for this intent."
            }

        total = sum(counter.values())
        df = self._load_raw_df()
        for next_intent, count in counter.most_common(limit):
            next_example = df[df["intent"].str.lower() == next_intent].head(1)
            if next_example.empty:
                continue
            row = next_example.iloc[0]
            suggestions.append({
                "next_intent": next_intent,
                "expected_query": row["instruction"],
                "sample_response": row["response"],
                "confidence": round(count / total, 2)
            })

        return {
            "predictions": suggestions,
            "confidence": round(sum(item["confidence"] for item in suggestions), 2)
        }

    def score_response_quality(self, generated_response: str, intent: str, sentiment: str = None) -> Dict[str, Any]:
        self._init_embeddings_and_vectorstore()
        filter_kwargs = {"intent": self._normalize_text(intent).lower()}
        if sentiment:
            filter_kwargs["sentiment"] = self._normalize_text(sentiment).lower()

        try:
            documents = self.vectorstore.similarity_search_with_relevance_scores(
                generated_response,
                k=5,
                filter=filter_kwargs
            )
        except Exception as e:
            logger.warning(f"Quality scoring retrieval failed: {e}")
            documents = []

        if documents:
            scores = [score for _, score in documents]
            avg_score = sum(scores) / len(scores)
            quality_score = round(min(max(avg_score, 0.0), 1.0) * 100, 2)
        else:
            quality_score = 50.0

        suggestions = []
        if quality_score < 70:
            suggestions.append(
                "Align your answer more closely with historical successful responses for this intent."
            )
        if sentiment and sentiment.lower() in ["angry", "frustrated"]:
            if not any(word in generated_response.lower() for word in ["sorry", "apologize", "understand", "empath", "thank you"]):
                suggestions.append(
                    "Add more empathy and acknowledge the customer's frustration explicitly."
                )
        if not suggestions:
            suggestions.append("Response quality matches historical patterns.")

        return {
            "quality_score": quality_score,
            "quality_suggestions": suggestions
        }

    def predict_escalation_risk(self, query_text: str, intent: str, sentiment: str = "neutral") -> Dict[str, Any]:
        df = self._load_raw_df()
        sentiment_key = self._normalize_text(sentiment).lower()
        intent_key = self._normalize_text(intent).lower()
        if df.empty:
            return {
                "risk_score": 0,
                "warning": "No dataset available to compute escalation patterns.",
                "intervention": "Monitor the conversation and escalate if sentiment worsens."
            }

        base_score = 0
        if sentiment_key in ["angry", "very_negative"]:
            base_score += 4
        elif sentiment_key in ["frustrated", "negative"]:
            base_score += 2

        escalation_examples = df[df["escalation_flag"] == True]
        matching_intent = df.iloc[0:0]
        if not escalation_examples.empty:
            matching_intent = escalation_examples[escalation_examples["intent"].str.lower() == intent_key]
            base_score += min(len(matching_intent), 4)

        if any(keyword in query_text.lower() for keyword in ["speak to", "human", "supervisor", "manager", "escalat", "refund", "complaint"]):
            base_score += 2

        risk_score = min(10, max(0, base_score))
        warning = ""
        intervention = ""
        if risk_score >= 8:
            warning = "High escalation risk detected for this customer query."
            intervention = "Offer an immediate human handoff or supervisor review."
        elif risk_score >= 5:
            warning = "Moderate escalation risk detected."
            intervention = "Use a very empathetic response and confirm the customer's issue clearly."
        else:
            warning = "Low escalation risk."
            intervention = "Proceed with a standard but empathetic answer."

        return {
            "risk_score": risk_score,
            "warning": warning,
            "intervention": intervention,
            "matched_escalation_examples": len(matching_intent) if df is not None else 0
        }

    def save_raw_data(self, rows: List[Dict[str, Any]], filename: str = "huggingface_raw_data.csv"):
        """Fetch data from Hugging Face datasets API"""
        url = f"https://datasets-server.huggingface.co/rows?dataset=bitext%2FBitext-customer-support-llm-chatbot-training-dataset&config=default&split=train&offset={offset}&length={length}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            rows = []
            for item in data.get('rows', []):
                row_data = item.get('row', {})
                rows.append({
                    'instruction': row_data.get('instruction', ''),
                    'category': row_data.get('category', ''),
                    'intent': row_data.get('intent', ''),
                    'response': row_data.get('response', ''),
                    'flags': row_data.get('flags', '')
                })
            
# logger.info(f"Fetched {len(rows)} records from Hugging Face (offset={offset})")
            return rows
        
        except Exception as e:
            # logger.info(f"❌ Error fetching from Hugging Face: {e}")
            return []

    def save_raw_data(self, rows: List[Dict[str, Any]], filename: str = "huggingface_raw_data.csv"):
        """Save raw Hugging Face data to CSV in data/raw_data."""
        if not rows:
            logger.warning("No raw rows to save to CSV")
            return

        csv_path = RAW_DATA_DIR / filename
        try:
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False, encoding="utf-8")
            logger.info(f"Saved raw data to {csv_path}")
        except Exception as e:
            logger.exception(f"Failed to save raw data to CSV: {e}")

    def has_vectorstore_data(self) -> bool:
        # Avoid initializing embeddings/vectorstore just to check for presence.
        persist_dir = Path(config.CHROMA_PERSIST_DIR)
        if not persist_dir.exists():
            return False
        # Heuristic: if the persist directory contains files, assume data exists.
        try:
            files = list(persist_dir.rglob("*"))
            return any(f.is_file() for f in files)
        except Exception:
            return False

    def load_raw_data_from_csv(self, max_records: int = None, filename: str = "huggingface_raw_data.csv") -> int:
        csv_path = RAW_DATA_DIR / filename
        if not csv_path.exists():
            logger.warning("No raw Hugging Face CSV exists to load.")
            return 0

        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except Exception as e:
            logger.exception(f"Failed to read raw CSV from {csv_path}: {e}")
            return 0

        rows = df.to_dict(orient="records")
        if max_records is not None:
            rows = rows[:max_records]

        documents = []
        for row in rows:
            content = f"""
            Customer Question: {row.get('instruction', '')}
            Category: {row.get('category', '')}
            Intent: {row.get('intent', '')}
            Support Response: {row.get('response', '')}
            """
            documents.append(Document(
                page_content=content,
                metadata=self._build_document_metadata(row, source="huggingface_csv")
            ))

        if documents:
            # Ensure embeddings/vectorstore exist before ingest
            self._init_embeddings_and_vectorstore()
            self.vectorstore.add_documents(documents)
            try:
                self.vectorstore.persist()
            except Exception:
                logger.warning("Chroma persistence is deprecated or unsupported by this version.")
            logger.info(f"Loaded {len(documents)} documents from raw CSV into ChromaDB")
            return len(documents)

        logger.info("No documents loaded from CSV")
        return 0

    def add_huggingface_to_vectorstore(self, max_records: int = 30000, force: bool = True, ingest: bool = True) -> int:
        """Fetch data from Hugging Face and add to vector store.

        If `ingest` is False, the method will only fetch the dataset rows and
        save them as CSV in `data/raw_data` without computing embeddings or
        adding documents to the vector store. This is useful when embedding
        model downloads are failing or should be deferred.
        """
        raw_csv = RAW_DATA_DIR / "huggingface_raw_data.csv"
        if raw_csv.exists() and not force:
            if self.has_vectorstore_data():
                # logger.info(
                #     "📌 Existing raw Hugging Face CSV found and vector store already has data. Skipping fetch. "
                #     "Use force=True to refresh data."
                # )
                return 0

            # logger.info(
            #     "📌 Existing raw Hugging Face CSV found but vector store appears empty. "
            #     "Loading data from CSV into vector store."
            # )
            if ingest:
                return self.load_raw_data_from_csv(max_records=max_records)
            logger.info("Ingest disabled; skipping load from CSV into vector store.")
            return 0

        all_rows = []
        offset = 0
        batch_size = 100
        
        # Fetch in batches until we reach max_records
        while len(all_rows) < max_records:
            rows = self.fetch_huggingface_data(offset=offset, length=batch_size)
            if not rows:
                break
            all_rows.extend(rows)
            offset += batch_size
            logger.info(f"Fetched {len(all_rows)} records so far...")
        
        # Convert to LangChain Documents
        documents = []
        for row in all_rows[:max_records]:
            # Create rich content combining instruction and response
            content = f"""
            Customer Question: {row['instruction']}
            Category: {row['category']}
            Intent: {row['intent']}
            Support Response: {row['response']}
            """
            
            doc = Document(
                page_content=content,
                metadata=self._build_document_metadata(row, source="huggingface")
            )
            documents.append(doc)
        
        # Save raw data to CSV before optionally adding to the vector store
        self.save_raw_data(all_rows)

        if not ingest:
            logger.info("Ingest disabled: raw CSV saved but embeddings not created.")
            return len(all_rows[:max_records])

        # Add to vector store
        if documents:
            # Ensure embeddings/vectorstore initialized
            self._init_embeddings_and_vectorstore()
            self.vectorstore.add_documents(documents)
            try:
                self.vectorstore.persist()
            except Exception:
                logger.warning("Chroma persistence is deprecated or unsupported by this version.")
            logger.info(f"Added {len(documents)} documents to ChromaDB")
            return len(documents)

        logger.info("No documents to add")
        return 0
    
    def fetch_and_save_csv_only(self, max_records: int = 30000) -> int:
        """Phase 1: Only fetch raw data and save to CSV. No embeddings."""
        raw_csv = RAW_DATA_DIR / "huggingface_raw_data.csv"
        if raw_csv.exists():
            logger.info("Raw CSV already exists. Skipping fetch.")
            return 0

        all_rows = []
        offset = 0
        batch_size = 100

        while len(all_rows) < max_records:
            rows = self.fetch_huggingface_data(offset=offset, length=batch_size)
            if not rows:
                break
            all_rows.extend(rows)
            offset += batch_size
            logger.info(f"Fetched {len(all_rows)} records so far...")

        if all_rows:
            self.save_raw_data(all_rows[:max_records])
            logger.info(f"Phase 1 complete: {len(all_rows[:max_records])} rows saved to CSV.")
            return len(all_rows[:max_records])

        logger.warning("No rows fetched from Hugging Face.")
        return 0

    def build_embeddings_from_csv(self, max_records: int = 30000) -> int:
        """Phase 2: Build embeddings from existing CSV. No fetching."""
        raw_csv = RAW_DATA_DIR / "huggingface_raw_data.csv"
        if not raw_csv.exists():
            logger.error("Cannot build embeddings: raw CSV not found. Run Phase 1 first.")
            return 0

        if self.has_vectorstore_data():
            logger.info("Vector store already has data. Skipping embedding creation.")
            return 0

        return self.load_raw_data_from_csv(max_records=max_records)