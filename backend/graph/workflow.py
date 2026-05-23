import json
from pathlib import Path
from datetime import datetime
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from backend.agents.intent_agent import IntentClassificationAgent
from backend.agents.sentiment_agent import SentimentPriorityAgent
from backend.agents.rag_agent import KnowledgeBaseRetrievalAgent
from backend.agents.response_agent import ResponseGenerationAgent
from backend.agents.escalation_agent import EscalationAgent
from backend.agents.analytics_agent import AnalyticsAgent
from backend.agents.qa_agent import QAComplianceAgent
from backend.utils.mlflow_tracer import MLflowTracer
import time
import uuid
from backend.utils.logger import get_logger

logger = get_logger(__name__)

class AgentState(TypedDict):
    conversation_id: str
    query: str
    intent_data: Dict[str, Any]
    sentiment_data: Dict[str, Any]
    rag_data: Dict[str, Any]
    response_data: Dict[str, Any]
    qa_data: Dict[str, Any]
    escalation_data: Dict[str, Any]
    final_response: str
    response_time_ms: float
    escalated: bool

class CustomerSupportWorkflow:
    def __init__(self):
        self.intent_agent = IntentClassificationAgent()
        self.sentiment_agent = SentimentPriorityAgent()
        self.rag_agent = KnowledgeBaseRetrievalAgent()
        self.response_agent = ResponseGenerationAgent()
        self.escalation_agent = EscalationAgent()
        self.analytics_agent = AnalyticsAgent()
        self.qa_agent = QAComplianceAgent()
        self.mlflow_tracer = MLflowTracer()
        
        self.workflow = self.build_workflow()
    
    def build_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("classify_intent", self.classify_intent)
        workflow.add_node("analyze_sentiment", self.analyze_sentiment)
        workflow.add_node("retrieve_knowledge", self.retrieve_knowledge)
        workflow.add_node("generate_response", self.generate_response)
        workflow.add_node("qa_validation", self.qa_validation)
        workflow.add_node("check_escalation", self.check_escalation)
        
        # Define edges
        workflow.set_entry_point("classify_intent")
        workflow.add_edge("classify_intent", "analyze_sentiment")
        workflow.add_edge("analyze_sentiment", "retrieve_knowledge")
        workflow.add_edge("retrieve_knowledge", "generate_response")
        workflow.add_edge("generate_response", "qa_validation")
        workflow.add_edge("qa_validation", "check_escalation")
        
        # Conditional routing from escalation
        workflow.add_conditional_edges(
            "check_escalation",
            self.should_escalate,
            {
                True: END,
                False: END
            }
        )
        
        compiled_workflow = workflow.compile()
        self.save_workflow_graph(compiled_workflow)
        return compiled_workflow

    def save_workflow_graph(self, compiled_workflow, output_file_path: str | None = None) -> None:
        output_path = Path(output_file_path or Path(__file__).resolve().parent / "workflow_graph.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        graph = compiled_workflow.get_graph()

        try:
            graph.draw_png(str(output_path))
            logger.info(f"Saved workflow graph image to {output_path}")
            return
        except Exception as exc:
            logger.warning(
                f"Graph.draw_png failed; attempting Mermaid PNG fallback: {exc}"
            )

        try:
            graph.draw_mermaid_png(output_file_path=str(output_path))
            logger.info(f"Saved workflow graph image via Mermaid fallback to {output_path}")
        except Exception as exc:
            logger.error(
                f"Unable to save workflow graph image to {output_path}: {exc}"
            )
    
    async def classify_intent(self, state: AgentState) -> AgentState:
        intent_data = await self._trace_agent_step(
            "classify_intent",
            self.intent_agent.classify,
            {"query": state["query"]},
            agent_name="IntentClassificationAgent",
            model_name=self._extract_model_name(self.intent_agent.llm),
        )
        state["intent_data"] = intent_data
        return state
    
    async def analyze_sentiment(self, state: AgentState) -> AgentState:
        sentiment_data = await self._trace_agent_step(
            "analyze_sentiment",
            self.sentiment_agent.analyze,
            {"query": state["query"], "intent": state["intent_data"].get("intent", "unknown")},
            agent_name="SentimentPriorityAgent",
            model_name=self._extract_model_name(self.sentiment_agent.llm),
        )
        state["sentiment_data"] = sentiment_data
        return state
    
    async def retrieve_knowledge(self, state: AgentState) -> AgentState:
        docs = await self._trace_agent_step(
            "retrieve_knowledge",
            self.rag_agent.retrieve_knowledge,
            {"query": state["query"], "intent": state["intent_data"].get("intent", "unknown")},
            agent_name="KnowledgeBaseRetrievalAgent",
            model_name=getattr(self.rag_agent.llm, "model", "unknown"),
        )
        
        rag_answer = await self._trace_agent_step(
            "answer_with_context",
            self.rag_agent.answer_with_context,
            {"query": state["query"], "intent": state["intent_data"].get("intent", "unknown"), "docs": docs},
            agent_name="KnowledgeBaseRetrievalAgent",
            model_name=self._extract_model_name(self.rag_agent.llm),
        )

        next_query = await self._trace_agent_step(
            "next_expected_query",
            self.rag_agent.get_next_expected_query,
            {"current_intent": state["intent_data"].get("intent", "unknown"), "current_sentiment": state["sentiment_data"].get("sentiment", "neutral")},
            agent_name="KnowledgeBaseRetrievalAgent",
            model_name=self._extract_model_name(self.rag_agent.llm),
        )
        rag_answer["next_expected_query"] = next_query
        rag_answer["retrieved_docs"] = docs
        state["rag_data"] = rag_answer
        return state
    
    async def generate_response(self, state: AgentState) -> AgentState:
        response_data = await self._trace_agent_step(
            "generate_response",
            self.response_agent.generate_response,
            {
                "query": state["query"],
                "intent": state["intent_data"].get("intent", "unknown"),
                "sentiment_data": state["sentiment_data"],
                "rag_data": state["rag_data"],
            },
            agent_name="ResponseGenerationAgent",
            model_name=self._extract_model_name(self.response_agent.llm),
        )
        state["response_data"] = response_data
        return state
    
    async def qa_validation(self, state: AgentState) -> AgentState:
        qa_data = await self._trace_agent_step(
            "qa_validation",
            self.qa_agent.validate_response,
            {
                "query": state["query"],
                "response": state["response_data"].get("response_text", ""),
                "rag_context": state["rag_data"].get("answer", ""),
            },
            agent_name="QAComplianceAgent",
            model_name=self._extract_model_name(self.qa_agent.llm),
        )
        state["qa_data"] = qa_data
        
        # If QA fails, modify response
        if not qa_data["is_compliant"]:
            state["response_data"]["response_text"] = (
                "I need to verify this information. Let me transfer you to a specialist.\n"
                f"Original response: {state['response_data']['response_text']}"
            )
            state["response_data"]["confidence"] = 0.3
        
        return state
    
    async def check_escalation(self, state: AgentState) -> AgentState:
        escalation_data = await self._trace_agent_step(
            "check_escalation",
            self.escalation_agent.should_escalate,
            {
                "query": state["query"],
                "intent": state["intent_data"].get("intent", "unknown"),
                "sentiment_data": state["sentiment_data"],
                "response_data": state["response_data"],
                "conversation_history": None,
            },
            agent_name="EscalationAgent",
            model_name="rule_engine",
        )
        
        state["escalation_data"] = escalation_data
        state["escalated"] = escalation_data["should_escalate"]
        
        if escalation_data["should_escalate"]:
            state["final_response"] = (
                f"I've escalated your issue to a human agent. "
                f"Ticket ID: {escalation_data['ticket']['ticket_id']}\n"
                f"Reason: {', '.join(escalation_data['reasons'])}\n\n"
                f"{state['response_data']['response_text']}"
            )
        else:
            state["final_response"] = state["response_data"]["response_text"]
        
        return state
    
    def should_escalate(self, state: AgentState) -> bool:
        return state["escalated"]

    def _extract_model_name(self, llm_obj: Any) -> str:
        """Extract model name from LLM object, supporting various LangChain LLM types."""
        if llm_obj is None:
            return "unknown"
        for attr in ["model", "model_name", "model_id", "_model_name"]:
            if hasattr(llm_obj, attr):
                val = getattr(llm_obj, attr, None)
                if val and isinstance(val, str):
                    return val
        return llm_obj.__class__.__name__

    def _summarize_payload(self, payload: Any, max_length: int = 160) -> str:
        try:
            text = json.dumps(payload, default=str)
        except Exception:
            text = str(payload)
        return text if len(text) <= max_length else text[:max_length].replace("\n", " ") + "..."

    async def _trace_agent_step(
        self,
        step_name: str,
        func,
        input_data: Dict[str, Any],
        agent_name: str,
        model_name: str,
    ) -> Any:
        timestamp = datetime.utcnow().isoformat() + "Z"
        start_time = time.time()
        
        # Prepare input data for logging - flatten dicts for better readability
        input_for_span = self._flatten_for_ui(input_data, max_depth=2)
        
        with self.mlflow_tracer.start_span(
            step_name,
            attributes={
                "agent.name": agent_name,
                "model.name": model_name,
                "step_name": step_name,
                "timestamp": timestamp,
            },
        ) as span:
            # Log input to span if span is available
            if span is not None:
                for key, value in input_for_span.items():
                    span.set_attribute(f"input.{key}", str(value)[:1000])
            
            result = await func(**input_data)
            latency_ms = (time.time() - start_time) * 1000
            
            # Prepare output data for logging
            output_for_span = self._flatten_for_ui(result, max_depth=2)
            
            # Log output to span if span is available
            if span is not None:
                for key, value in output_for_span.items():
                    span.set_attribute(f"output.{key}", str(value)[:1000])
                span.set_attribute("execution_time_ms", round(latency_ms, 2))
            
            # Log metrics
            self.mlflow_tracer.log_metric("latency_ms", latency_ms)
            
            # Log tags with model name, input and output information
            self.mlflow_tracer.log_tag("agent.name", agent_name)
            self.mlflow_tracer.log_tag("model.name", model_name)
            self.mlflow_tracer.log_tag("agent.step", step_name)
            self.mlflow_tracer.log_tag("step.latency_ms", f"{round(latency_ms, 2)}")
            
            # Log agent-specific important fields
            self._log_agent_specific_data(step_name, agent_name, result)
            
            # Log full input/output as JSON artifacts for detailed inspection
            self.mlflow_tracer.log_json_artifact(
                {
                    "input": input_data,
                    "output": result,
                    "latency_ms": round(latency_ms, 2),
                    "timestamp": timestamp,
                    "agent_name": agent_name,
                    "model_name": model_name,
                    "step_name": step_name,
                },
                filename=f"{step_name}_trace_{int(start_time * 1000)}.json",
            )
            
            # Also log input and output as separate artifacts for easy reference
            self.mlflow_tracer.log_json_artifact(
                input_data,
                filename=f"{step_name}_input.json",
            )
            self.mlflow_tracer.log_json_artifact(
                result,
                filename=f"{step_name}_output.json",
            )
            
            return result
    
    def _flatten_for_ui(self, data: Any, max_depth: int = 2, current_depth: int = 0, prefix: str = "") -> Dict[str, Any]:
        """Flatten nested data structures for UI display without excessive nesting."""
        flattened = {}
        
        if current_depth >= max_depth or not isinstance(data, dict):
            return {prefix or "value": str(data)[:1000]}
        
        for key, value in data.items():
            new_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict) and current_depth < max_depth - 1:
                flattened.update(self._flatten_for_ui(value, max_depth, current_depth + 1, new_key))
            elif isinstance(value, (list, tuple)):
                if value and isinstance(value[0], dict) and current_depth < max_depth - 1:
                    flattened[f"{new_key}_count"] = len(value)
                    if value:
                        flattened.update(self._flatten_for_ui(value[0], max_depth, current_depth + 1, f"{new_key}[0]"))
                else:
                    flattened[f"{new_key}_count"] = len(value)
                    flattened[new_key] = str(value)[:1000]
            else:
                flattened[new_key] = str(value)[:1000]
        
        return flattened

    def _log_agent_specific_data(self, step_name: str, agent_name: str, result: Any) -> None:
        """Log important fields from agent outputs for better visibility in MLflow UI."""
        if not isinstance(result, dict):
            return
        
        try:
            if agent_name == "EscalationAgent" or "escalation" in step_name.lower():
                self.mlflow_tracer.log_tag("escalation.should_escalate", str(result.get("should_escalate", False)))
                reasons = result.get("reasons", [])
                if reasons:
                    self.mlflow_tracer.log_tag("escalation.reasons", ", ".join(reasons)[:1000])
                ticket = result.get("ticket", {})
                if ticket:
                    self.mlflow_tracer.log_tag("escalation.ticket_id", ticket.get("ticket_id", "N/A"))
                self.mlflow_tracer.log_tag("escalation.priority", result.get("priority", "N/A"))
                self.mlflow_tracer.log_tag("escalation.risk_score", str(result.get("risk_score", 0)))
                
            elif agent_name == "SentimentPriorityAgent" or "sentiment" in step_name.lower():
                self.mlflow_tracer.log_tag("sentiment.value", result.get("sentiment", "unknown"))
                self.mlflow_tracer.log_tag("sentiment.priority_score", str(result.get("priority_score", 0)))
                self.mlflow_tracer.log_tag("sentiment.escalation_needed", str(result.get("escalation_needed", False)))
                emotions = result.get("emotions", [])
                if emotions:
                    self.mlflow_tracer.log_tag("sentiment.emotions", ", ".join(emotions))
                self.mlflow_tracer.log_tag("sentiment.reason", result.get("reason", "")[:1000])
                
            elif agent_name == "KnowledgeBaseRetrievalAgent":
                if "retrieve_knowledge" in step_name.lower():
                    if isinstance(result, list):
                        self.mlflow_tracer.log_tag("retrieval.num_docs", str(len(result)))
                        if result:
                            first_doc_summary = result[0].get("content", "")[:1000] if isinstance(result[0], dict) else ""
                            self.mlflow_tracer.log_tag("retrieval.first_doc_preview", first_doc_summary)
                            
                elif "answer_with_context" in step_name.lower():
                    answer = result.get("answer", "")
                    if answer:
                        self.mlflow_tracer.log_tag("answer.text", answer[:1000])
                    self.mlflow_tracer.log_tag("answer.num_docs_used", str(result.get("num_docs", 0)))
                    
                elif "next_expected_query" in step_name.lower():
                    next_query = result.get("next_query", "")
                    if next_query:
                        self.mlflow_tracer.log_tag("next_query.value", next_query[:1000])
                        
            elif agent_name == "ResponseGenerationAgent" or "generate_response" in step_name.lower():
                response_text = result.get("response_text", "")
                if response_text:
                    self.mlflow_tracer.log_tag("response.text", response_text[:1000])
                self.mlflow_tracer.log_tag("response.confidence", str(result.get("confidence", 0)))
                self.mlflow_tracer.log_tag("response.quality_score", str(result.get("quality_score", 0)))
                
            elif agent_name == "QAComplianceAgent" or "qa" in step_name.lower():
                self.mlflow_tracer.log_tag("qa.is_compliant", str(result.get("is_compliant", False)))
                issues = result.get("issues", [])
                if issues:
                    self.mlflow_tracer.log_tag("qa.issues", ", ".join(issues)[:1000])
                self.mlflow_tracer.log_tag("qa.score", str(result.get("score", 0)))
                    
            elif agent_name == "IntentClassificationAgent" or "intent" in step_name.lower():
                intent = result.get("intent", "")
                if intent:
                    self.mlflow_tracer.log_tag("intent.classified", intent)
                self.mlflow_tracer.log_tag("intent.confidence", str(result.get("confidence", 0)))
                
        except Exception as e:
            logger.debug(f"Error logging agent-specific data: {e}")

    async def process_query(self, query: str, session_id: str | None = None) -> Dict[str, Any]:
        start_time = time.time()
        logger.info(f"process_query start: query={query[:120]}")
        conversation_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat() + "Z"

        with self.mlflow_tracer.start_run(
            run_name=f"conversation_{conversation_id}",
            nested=False,
            conversation_id=conversation_id,
            session_id=session_id,
            query=query,
            timestamp=timestamp,
        ):
            with self.mlflow_tracer.start_span(
                "conversation",
                attributes={
                    "conversation_id": conversation_id,
                    "session_id": session_id or "anonymous",
                    "query": query,
                    "timestamp": timestamp,
                },
            ):
                initial_state = {
                    "conversation_id": conversation_id,
                    "query": query,
                    "intent_data": {},
                    "sentiment_data": {},
                    "rag_data": {},
                    "response_data": {},
                    "qa_data": {},
                    "escalation_data": {},
                    "final_response": "",
                    "response_time_ms": 0,
                    "escalated": False,
                }

                final_state = await self.workflow.ainvoke(initial_state)
                response_time_ms = (time.time() - start_time) * 1000
                final_state["response_time_ms"] = response_time_ms
                self.mlflow_tracer.log_metric("response_time_ms", response_time_ms)
                self.mlflow_tracer.log_tag("conversation.final_response", final_state["final_response"])
                self.mlflow_tracer.log_json_artifact(
                    {
                        "final_state": final_state,
                        "response_time_ms": round(response_time_ms, 2),
                    },
                    filename=f"conversation_{conversation_id}_complete.json",
                )

                analytics_data = {
                    "conversation_id": final_state["conversation_id"],
                    "query": query,
                    "intent": final_state["intent_data"].get("intent"),
                    "sentiment": final_state["sentiment_data"].get("sentiment"),
                    "priority_score": final_state["sentiment_data"].get("priority_score"),
                    "quality_score": final_state["response_data"].get("quality_score"),
                    "escalated": final_state["escalated"],
                    "response_time_ms": response_time_ms,
                    "timestamp": time.time(),
                }
                self.analytics_agent.track_conversation(analytics_data)
                logger.info(f"process_query completed: conversation_id={final_state['conversation_id']} response_time_ms={response_time_ms:.2f}ms")

                return {
                    "conversation_id": final_state["conversation_id"],
                    "response": final_state["final_response"],
                    "intent": final_state["intent_data"],
                    "sentiment": final_state["sentiment_data"],
                    "qa_results": final_state["qa_data"],
                    "escalation": final_state["escalation_data"],
                    "next_expected_query": final_state["rag_data"].get("next_expected_query"),
                    "response_time_ms": response_time_ms,
                    "confidence": final_state["response_data"].get("confidence", 0.5),
                    "quality_score": final_state["response_data"].get("quality_score"),
                }
