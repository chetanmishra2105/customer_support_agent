# AI Customer Support Orchestration Platform

## Enterprise Multi-Agent Customer Support System

### Features

- **7 Specialized AI Agents**: Intent, Sentiment, RAG, Response, Escalation, Analytics, QA
- **LangGraph Workflow**: Orchestrated agent communication
- **RAG Architecture**: Knowledge base retrieval with ChromaDB
- **Real-time Analytics**: Support trends, SLA monitoring, escalation tracking
- **Enterprise Features**: PII detection, hallucination prevention, policy compliance

### Quick Start

1. **Setup Environment**
```bash
cp .env.example .env
# Add your GROQ_API_KEY and any other required secret values to .env
```

2. **Secure secret management**
- Keep `.env` out of Git; this repo already ignores `.env` in `.gitignore`.
- Store only non-sensitive defaults in `.env.example`.
- Use GitHub Secrets for CI/CD values like `DOCKER_USERNAME` and `DOCKER_PASSWORD`.

3. **Workflow / Docker Hub**
- The GitHub Actions workflow builds backend and frontend images.
- It logs in to Docker Hub using secrets and pushes:
  - `${{ secrets.DOCKER_USERNAME }}/customer-support-backend:latest`
  - `${{ secrets.DOCKER_USERNAME }}/customer-support-frontend:latest`

4. **Run locally**
```bash
streamlit run frontend/app.py
# and in another terminal:
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```
