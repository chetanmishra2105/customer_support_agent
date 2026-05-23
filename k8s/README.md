# Kubernetes Deployment for Customer Support Agent

## Overview
This directory contains a sample Kubernetes manifest for deploying the app locally with Docker Desktop Kubernetes.

## What is included
- Namespace: `customer-support`
- Secrets: `customer-support-secrets`
- ConfigMap: `customer-support-config`
- Deployments and Services for:
  - `redis`
  - `chromadb`
  - `backend`
  - `frontend`
- PersistentVolumeClaims for Redis and ChromaDB data

## Steps to deploy
1. Enable Kubernetes in Docker Desktop.
2. From the repo root, apply the manifest:
   ```bash
   kubectl apply -f k8s/manifest.yml
   ```
3. Confirm pods are running:
   ```bash
   kubectl get pods -n customer-support
   ```
4. Confirm services:
   ```bash
   kubectl get svc -n customer-support
   ```
5. Open the frontend locally:
   ```bash
   kubectl port-forward svc/frontend 8501:8501 -n customer-support
   ```
   Then open `http://localhost:8501` in your browser.

## Updating secrets
The manifest includes placeholder secret values.
Replace them with real secrets before deploying, or use this command instead:

```bash
kubectl create secret generic customer-support-secrets \
  --namespace=customer-support \
  --from-literal=GROQ_API_KEY="YOUR_GROQ_API_KEY" \
  --from-literal=LANGCHAIN_API_KEY="YOUR_LANGCHAIN_API_KEY" \
  --from-literal=SECRET_KEY="YOUR_SECRET_KEY"
```

Then apply the manifest again.

## Important notes
- In Kubernetes, services talk to each other by name, not `localhost`.
- The frontend uses `BACKEND_URL=http://backend:8000`.
- Use `ConfigMap` for non-sensitive settings and `Secret` for API keys.
- If you modify images, rebuild and push them to a registry or make them available to your Kubernetes cluster.

## Cleanup
To remove the deployment:
```bash
kubectl delete -f k8s/manifest.yml
```
