# Model Deployment Guide (Local → AWS)

## 0) Repo Layout (example)
```
.
├─ app/                   # your python package / server code
│  ├─ main.py             # starts the API (uvicorn/flask run)
│  ├─ model.py            # model load + predict
│  └─ ...                 # utils, checkpoints, etc.
├─ checkpoints/           # model weights (if bundled)
├─ requirements.txt
├─ Dockerfile
├─ Makefile               # optional helpers
└─ README.md
```

> ✅ Your API should expose:
> - `GET /health` → returns `{ "status": "ok" }` and/or `model_loaded=true`
> - `POST /predict` → accepts JSON (e.g., `{"input":"..."}`) and returns JSON or an image

---

## 1) Local Development

### 1.1 Create & activate environment
```bash
# Conda
conda create -n model-deploy python=3.11 -y
conda activate model-deploy

# OR venv
python -m venv .venv
source .venv/bin/activate
```

### 1.2 Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 1.3 Required environment variables
Add a `.env.local` (optional) or export in shell:
```bash
export MODEL_PATH="app/checkpoints/best_model.pth"   # adjust to your weights
export VOCAB_JSON="app/token_to_idx.json"            # optional
export PROP_HEAD_PATH="app/checkpoints/prop_head.pt" # optional
export KMP_DUPLICATE_LIB_OK="TRUE"                   # (if PyTorch/OMP conflict)
```

### 1.4 Run the API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 1.5 Smoke tests (no bash UI needed)
- Swagger UI: `http://127.0.0.1:8000/docs`
- Health: `curl -s http://127.0.0.1:8000/health`
- Predict (JSON):  
  ```bash
  curl -s -X POST http://127.0.0.1:8000/predict     -H "Content-Type: application/json"     -d '{"smiles":"CCO"}'
  ```

---

## 2) Containerization (Docker)

### 2.1 Minimal Dockerfile (Python API)
```dockerfile
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# System deps (add as needed)
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
```

### 2.2 Build & run locally
```bash
docker build -t model-api:local .
docker run --rm -p 8000:8000   -e MODEL_PATH="app/checkpoints/best_model.pth"   -e VOCAB_JSON="app/token_to_idx.json"   -e PROP_HEAD_PATH="app/checkpoints/prop_head.pt"   -e KMP_DUPLICATE_LIB_OK="TRUE"   model-api:local
```

Test again at `http://127.0.0.1:8000/docs`.

---

## 3) AWS Deployment (ECR → App Runner)

### 3.1 Prereqs
- AWS account + payment method
- AWS CLI configured (`aws configure`)
- Region: `us-east-1`
- IAM user with:
  - `AmazonEC2ContainerRegistryFullAccess`
  - `AWSAppRunnerFullAccess`
  - `CloudWatchLogsFullAccess`

### 3.2 Create (or reuse) an ECR repo & push the image
```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_REPO=model-api
aws ecr describe-repositories --repository-names $ECR_REPO --region $AWS_REGION >/dev/null 2>&1 ||   aws ecr create-repository --repository-name $ECR_REPO --region $AWS_REGION

aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

docker tag model-api:local ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:v1
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:v1
```

### 3.3 Create App Runner service (Console)
1. **App Runner → Create service → Container image**
2. **Source**: Amazon ECR → pick repo `${ECR_REPO}`, tag `v1`
3. **Port**: `8000`
4. **Runtime environment variables**:
   ```
   MODEL_PATH=app/checkpoints/best_model.pth
   VOCAB_JSON=app/token_to_idx.json
   PROP_HEAD_PATH=app/checkpoints/prop_head.pt
   KMP_DUPLICATE_LIB_OK=TRUE
   ```
5. **Health check path**: `/health`
6. **Compute**: 1 vCPU / 2 GB
7. **Deployment settings**: Manual (you redeploy with the Deploy button)
8. **ECR access role**: select/create AppRunnerECRAccessRole
9. Create → wait until Running

### 3.4 Test the public endpoint
- Copy Default domain from the service page:
  - Health: `https://<domain>/health`
  - Swagger: `https://<domain>/docs`
  - Predict:
    ```bash
    curl -s -X POST https://<domain>/predict       -H "Content-Type: application/json"       -d '{"smiles":"CCO"}'
    ```

### 3.5 Update workflow
- Rebuild & push a new tag (e.g., v2)
- In App Runner service page, switch the image tag to v2 (or hit Deploy if auto pickup is enabled)

### 3.6 Control costs
- When done, Pause the service (no compute charges), or Delete it to stop all charges.
- ECR image storage is tiny; delete the image/repo if you want $0.

---

## 4) Troubleshooting

- **500 Internal Server Error**
  - Check CloudWatch Logs
  - Confirm env vars exist and paths match files in the image

- **Model not loaded on /health**
  - Add lazy-load or warmup in startup

- **RDKit/native lib issues**
  - Use micromamba/conda base and install from conda-forge

- **Access Key needs subscription**
  - New accounts may need activation by AWS Support

---

## 5) Alternatives (free/cheap)
- Render (free tier)
- Railway (free credits)
- Google Cloud Run (free quota)
- Hugging Face Spaces (Gradio demos)

---

## 6) Security & Cost Hygiene
- Never commit secrets; use env vars / Secrets Manager
- Set Zero-Spend budget
- Pause/Delete unused services
- Keep images small
