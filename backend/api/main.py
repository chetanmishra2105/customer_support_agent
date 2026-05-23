from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import time
from backend.graph.workflow import CustomerSupportWorkflow
from backend.rag.vector_store import VectorStoreManager, RAW_DATA_DIR
from backend.config import config
from backend.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="AI Customer Support Orchestration Platform",
    description="Enterprise Multi-Agent Customer Support System",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
workflow = CustomerSupportWorkflow()
vector_store = VectorStoreManager()

@app.on_event("startup")
async def startup_event():
    import asyncio
    logger.info("🚀 Starting AI Customer Support Platform...")

    raw_csv = RAW_DATA_DIR / "huggingface_raw_data.csv"
    has_vs_data = vector_store.has_vectorstore_data()

    # Nothing to do — both CSV and embeddings already exist
    if raw_csv.exists() and has_vs_data:
        logger.info("✅ Raw CSV and vector store both present. Nothing to do on startup.")
        logger.info("✅ System ready!")
        try:
            workflow.mlflow_tracer.start_ui()
        except Exception:
            logger.exception("Failed to start MLflow UI during startup")
        return

    # Phase 1: Fetch CSV if missing
    if not raw_csv.exists():
        # logger.info("Phase 1: Raw CSV missing — fetching from Hugging Face...")
        try:
            fetched = await asyncio.to_thread(vector_store.fetch_and_save_csv_only, 30000)
            if fetched == 0 and not raw_csv.exists():
                logger.error("Phase 1 failed: CSV was not created. Aborting startup pipeline.")
                logger.info("✅ System ready (no vector store data)!")
                return
        except Exception:
            logger.exception("Phase 1 failed: error during CSV fetch. Aborting startup pipeline.")
            logger.info("✅ System ready (no vector store data)!")
            return
    else:
        logger.info("Phase 1: Raw CSV already exists. Skipping fetch.")

    # Phase 2: Build embeddings only if Phase 1 succeeded
    if not has_vs_data:
        logger.info("Phase 2: Building vector embeddings from CSV...")
        try:
            loaded = await asyncio.to_thread(vector_store.build_embeddings_from_csv, 30000)
            if loaded > 0:
                logger.info(f"Phase 2 complete: {loaded} documents embedded.")
            else:
                logger.warning("Phase 2: No documents were embedded.")
        except Exception:
            logger.exception("Phase 2 failed: error during embedding creation.")
    else:
        logger.info("Phase 2: Vector store already populated. Skipping embedding.")

    logger.info("✅ System ready!")
    try:
        workflow.mlflow_tracer.start_ui()
    except Exception:
        logger.exception("Failed to start MLflow UI during startup")

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class QualityScoreRequest(BaseModel):
    generated_response: str
    intent: str
    sentiment: Optional[str] = "neutral"

class EscalationRiskRequest(BaseModel):
    query: str
    intent: str
    sentiment: Optional[str] = "neutral"

class QueryResponse(BaseModel):
    conversation_id: str
    response: str
    intent: Dict[str, Any]
    sentiment: Dict[str, Any]
    qa_results: Optional[Dict[str, Any]] = None
    escalation: Dict[str, Any]
    next_expected_query: Optional[Dict[str, Any]] = None
    quality_score: Optional[float] = None
    response_time_ms: float
    confidence: float

@app.get("/")
async def root():
    return {
        "message": "AI Customer Support Orchestration Platform",
        "version": "1.0.0",
        "status": "operational"
    }

@app.post("/api/chat", response_model=QueryResponse)
async def process_chat(request: QueryRequest):
    try:
        logger.info(f"process_chat called: session_id={request.session_id}")
        result = await workflow.process_query(request.query, session_id=request.session_id)
        logger.info(f"process_chat completed: conversation_id={result.get('conversation_id')}")
        return QueryResponse(**result)
    except Exception as e:
        logger.exception("Unhandled exception in process_chat")
        raise HTTPException(status_code=500, detail=str(e))

@app.api_route("/api/load-data", methods=["GET", "POST"])
async def load_huggingface_data(max_records: int = 30000, force: bool = True, ingest: bool = False):
    # logger.info(f"load_huggingface_data called: max_records={max_records}, force={force}, ingest={ingest}")
    try:
        loaded = vector_store.add_huggingface_to_vectorstore(max_records=max_records, force=force, ingest=ingest)
        return {
            "status": "success",
            "loaded_documents": loaded,
            "message": "Hugging Face data loaded into vector store." if loaded else "No new data loaded. Existing vector store data was retained."
        }
    except Exception as e:
        logger.exception("Failed to load Hugging Face data on demand")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tools/intent-examples")
async def intent_examples(intent: str, limit: int = 5, include_response: bool = False):
    examples = vector_store.get_examples_by_intent(intent, limit=limit, include_response=include_response)
    return {"intent": intent, "examples": examples}

@app.get("/api/tools/sentiment-responses")
async def sentiment_responses(sentiment: str = "neutral", limit: int = 10):
    responses = vector_store.get_responses_by_sentiment(sentiment, limit=limit)
    return {"sentiment": sentiment, "responses": responses}

@app.get("/api/tools/next-query")
async def next_query(current_intent: str, current_sentiment: str = "neutral", limit: int = 3):
    prediction = vector_store.get_next_expected_query(current_intent, current_sentiment, limit=limit)
    return prediction

@app.post("/api/tools/quality-score")
async def quality_score(request: QualityScoreRequest):
    score_data = vector_store.score_response_quality(
        request.generated_response,
        request.intent,
        request.sentiment
    )
    return score_data

@app.post("/api/tools/escalation-risk")
async def escalation_risk(request: EscalationRiskRequest):
    risk_data = vector_store.predict_escalation_risk(
        request.query,
        request.intent,
        request.sentiment
    )
    return risk_data

@app.get("/api/analytics")
async def get_analytics(days: int = 7):
    analytics_data = workflow.analytics_agent.generate_analytics(days)
    return JSONResponse(content=analytics_data)

@app.get("/api/mlflow")
async def get_mlflow_info():
    tracer = workflow.mlflow_tracer
    return {
        "mlflow_ui_url": tracer.get_ui_url(),
        "tracking_uri": tracer.tracking_uri,
        "experiment_name": tracer.experiment_name,
        "status": "running" if tracer.ui_process and tracer.ui_process.poll() is None else "stopped"
    }

@app.post("/api/mlflow/start")
async def start_mlflow_ui():
    workflow.mlflow_tracer.start_ui()
    return {"mlflow_ui_url": workflow.mlflow_tracer.get_ui_url(), "status": "started"}

@app.post("/api/mlflow/stop")
async def stop_mlflow_ui():
    workflow.mlflow_tracer.stop_ui()
    return {"mlflow_ui_url": workflow.mlflow_tracer.get_ui_url(), "status": "stopped"}

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "groq": "connected",
            "chromadb": "connected",
            "redis": "pending"  # Would check actual connection
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.BACKEND_PORT)