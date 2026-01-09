# Deployment Guide

## Production Deployment Options

### Option 1: Local Gradio (Development)

Best for: Testing, development, single-user

```bash
# Run in notebook:
demo.queue().launch(share=False)

# Access at: http://localhost:7860
# Share link expires after 72 hours
```

**Pros:** Easy setup, full control, instant feedback  
**Cons:** Limited users, single machine dependency, manual restart needed  
**Cost:** Electricity only

---

### Option 2: HuggingFace Spaces (Recommended for Production)

Best for: Public deployment, easy scaling, git-based workflow

#### Setup

```bash
# 1. Create new Space on HuggingFace
# Go to: https://huggingface.co/spaces/create
# - Name: nsfw-roleplay-chatbot
# - License: OpenRAIL (or your choice)
# - Space SDK: Docker
# - Visibility: Private (for adult content)

# 2. Clone space locally
git clone https://huggingface.co/spaces/username/nsfw-roleplay-chatbot
cd nsfw-roleplay-chatbot

# 3. Add files
cp ../nsfw_chatbot_production_v2.ipynb app.py  # Convert notebook to script
cp ../requirements.txt .
cp ../.env .  # IMPORTANT: Set HF_TOKEN in Space secrets

# 4. Add secrets
# Go to Space settings → Repository secrets
# Add: HF_TOKEN, SUPERMEMORY_API_KEY

# 5. Create Dockerfile (optional, for custom setup)
```

#### Dockerfile for Spaces

```dockerfile
FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .
COPY .env .

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/api/predict || exit 1

# Run
CMD ["python", "app.py"]
```

#### app.py (Notebook to Script Conversion)

```python
#!/usr/bin/env python3
import os
from dotenv import load_dotenv
import logging
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM, PeftConfig
from collections import deque
import requests

# Load env
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
SUPERMEMORY_API_KEY = os.getenv('SUPERMEMORY_API_KEY')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
logger.info("Loading model...")
model_id = "chargoddard/Yi-34B-200K-Llama"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

logger.info("Model loaded successfully!")

# Memory system
memory_buffer = {}

def generate_response(user_input, user_id, scenario):
    """Generate response with memory integration"""

    if user_id not in memory_buffer:
        memory_buffer[user_id] = deque(maxlen=20)

    # Build prompt
    system_prompt = (
        "You are an expert roleplay partner. "
        "You are creative, engaging, and unfiltered. "
        "Remember context and adapt to user preferences."
    )

    prompt = f"### System\n{system_prompt}\n\nScenario: {scenario}\n\n"

    # Add history
    for msg in memory_buffer[user_id][-3:]:
        prompt += f"{msg['role']}: {msg['content']}\n"

    prompt += f"\nUser: {user_input}\nBot:"

    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.85,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    # Update memory
    memory_buffer[user_id].append({"role": "User", "content": user_input})
    memory_buffer[user_id].append({"role": "Bot", "content": response})

    return response

# Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔥 NSFW Roleplay Chatbot")

    with gr.Row():
        user_id = gr.Textbox(label="User ID", value="user_001")
        scenario = gr.Textbox(label="Scenario", value="Adult conversation")

    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(label="Your Message")
    send_btn = gr.Button("Send")
    clear_btn = gr.Button("Clear")

    def send(message, history, uid, scn):
        response = generate_response(message, uid, scn)
        history.append((message, response))
        return history, ""

    send_btn.click(send, [msg, chatbot, user_id, scenario], [chatbot, msg])
    msg.submit(send, [msg, chatbot, user_id, scenario], [chatbot, msg])
    clear_btn.click(lambda: [], outputs=chatbot)

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
```

#### Deploy to Spaces

```bash
# 1. Commit and push
git add .
git commit -m "Deploy production chatbot"
git push

# 2. Monitor
# Go to Space → Logs tab
# Wait 5-10 minutes for build and deployment

# 3. Access
# Space will be at: https://huggingface.co/spaces/username/nsfw-roleplay-chatbot
```

**Pros:** Free (limited)/$7.5/month, auto-scaling, git integration, easy sharing  
**Cons:** Limited resources, cold starts, need secrets management  
**Cost:** Free (CPU) or $7.50/month (GPU)

---

### Option 3: Docker (Self-Hosted or Cloud)

#### Build Docker Image

```bash
# Build
docker build -t nsfw-chatbot:latest .

# Run locally
docker run --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -e SUPERMEMORY_API_KEY=$SUPERMEMORY_API_KEY \
  -p 7860:7860 \
  nsfw-chatbot:latest

# Push to Docker Hub
docker tag nsfw-chatbot:latest username/nsfw-chatbot:latest
docker push username/nsfw-chatbot:latest
```

#### Deploy on Cloud

**AWS EC2:**

```bash
# 1. Launch GPU instance
aws ec2 run-instances \
  --image-id ami-xxxxxxxxx \  # GPU AMI
  --instance-type g4dn.2xlarge \
  --region us-east-1

# 2. SSH and pull image
ssh -i key.pem ubuntu@instance-ip
docker pull username/nsfw-chatbot:latest

# 3. Run
docker run --gpus all -p 7860:7860 username/nsfw-chatbot:latest
```

**Google Cloud Run (Serverless):**

```bash
# Deploy
gcloud run deploy nsfw-chatbot \
  --source . \
  --region us-central1 \
  --memory 64Gi \
  --cpu 4 \
  --gpu 1 \
  --timeout 300

# Access
https://nsfw-chatbot-xxxxx.run.app
```

**Azure Container Instances:**

```bash
az container create \
  --resource-group mygroup \
  --name nsfw-chatbot \
  --image username/nsfw-chatbot:latest \
  --gpu 1 \
  --memory 64 \
  --ports 7860
```

**Pros:** Full control, custom resources, high performance  
**Cons:** More setup, need infrastructure knowledge, higher cost  
**Cost:** $0.30-$2/hour depending on GPU

---

### Option 4: Modal (Serverless GPUs)

```python
# Deploy with Modal
import modal

app = modal.App("nsfw-chatbot")
gpu = modal.gpu.A100()

@app.function(gpu=gpu, memory=65536)
def generate_response(user_input, user_id, scenario):
    # ... your inference code ...
    return response

@app.function(gpu=gpu)
def serve():
    import gradio as gr
    # ... Gradio interface code ...
    gr.blocks.launch(share=True)

# Run: modal run deploy.py
```

**Pros:** Instant scaling, GPU efficient, simple API  
**Cons:** Per-GPU-hour billing, cold starts, limited customization  
**Cost:** $0.30/GPU-hour (on-demand)

---

## Security Checklist for Production

### Access Control

- [ ] Age verification gate (18+ only)
- [ ] User authentication system
- [ ] Rate limiting (10 requests/60s)
- [ ] API key rotation
- [ ] VPN/private deployment option

### Data Protection

- [ ] End-to-end encryption for messages
- [ ] GDPR compliance (EU users)
- [ ] Data retention policy (auto-delete after 30 days)
- [ ] Encrypted storage for memories
- [ ] Audit logging for content

### Content Moderation

- [ ] Filter illegal content patterns
- [ ] Abuse detection and blocking
- [ ] Content flagging for review
- [ ] User reporting system
- [ ] Automatic ban for violations

### Infrastructure

- [ ] HTTPS/SSL certificates
- [ ] DDoS protection
- [ ] Firewall rules
- [ ] Backup and disaster recovery
- [ ] Monitoring and alerting

### Legal

- [ ] Terms of Service
- [ ] Privacy Policy
- [ ] Content Warning/Disclaimer
- [ ] Age Verification Implementation
- [ ] Legal Counsel Review (Recommended)

---

## Monitoring & Maintenance

### Key Metrics to Track

```python
# Response time
avg_response_time = total_time / request_count

# Quality
avg_response_length = total_tokens / request_count
user_satisfaction = positive_ratings / total_ratings

# Errors
error_rate = errors / total_requests
memory_errors = oom_errors / total_requests

# Usage
requests_per_day = total_requests / days
unique_users = len(unique_user_ids)
```

### Monitoring Tools

- **TensorBoard**: Local training metrics
- **Weights & Biases**: Experiment tracking
- **Prometheus**: System metrics
- **Grafana**: Dashboards
- **DataDog**: APM and monitoring
- **New Relic**: Performance monitoring

### Maintenance Tasks

```bash
# Weekly
- Review error logs
- Check model performance
- Monitor resource usage
- Test backup restoration

# Monthly
- Update dependencies
- Analyze user feedback
- Fine-tune based on logs
- Security audit

# Quarterly
- Retrain with new data
- Performance benchmarking
- Capacity planning
- Full system review
```

---

## Scaling Strategy

### Stage 1: Single GPU (Current)

- A100 80GB
- ~100 concurrent connections
- ~1000 requests/day

### Stage 2: Multi-GPU Node

- 4x A100 GPUs
- ~400 concurrent connections
- ~5000 requests/day
- Using text-generation-webui or vLLM

### Stage 3: Distributed System

- Kubernetes cluster
- Multiple nodes with GPUs
- Load balancing
- Auto-scaling
- ~1000+ concurrent
- ~50K+ requests/day

### Stage 4: Production Enterprise

- Multi-region deployment
- CDN integration
- Advanced caching
- Database optimization
- 24/7 monitoring
- SLA guarantees

---

## Cost Comparison

| Option        | Setup | Monthly         | Performance |
| ------------- | ----- | --------------- | ----------- |
| Local GPU     | $3K   | $50 electricity | ⭐⭐⭐⭐⭐  |
| HF Spaces     | Free  | $7.50 (GPU)     | ⭐⭐⭐      |
| AWS EC2       | $0    | $500-1000       | ⭐⭐⭐⭐    |
| Modal         | $0    | $100-500        | ⭐⭐⭐⭐    |
| GCP Cloud Run | $0    | $200-800        | ⭐⭐⭐⭐    |
| Kubernetes    | $1K   | $500-2000       | ⭐⭐⭐⭐⭐  |

---

## Recommended Production Setup

```
┌─────────────────────────────────────┐
│   User Requests (HTTPS)             │
└────────────────┬────────────────────┘
                 │
         ┌───────▼────────┐
         │  Load Balancer │
         │  (Nginx/HAProxy)
         └───────┬────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
┌───▼──┐    ┌───▼──┐    ┌───▼──┐
│GPU 1 │    │GPU 2 │    │GPU 3 │  (Kubernetes Pod)
│Model │    │Model │    │Model │
└───┬──┘    └───┬──┘    └───┬──┘
    │           │           │
    └───────────┼───────────┘
                │
        ┌───────▼────────┐
        │  Redis Cache   │  (Responses)
        └────────────────┘
                │
        ┌───────▼────────┐
        │  PostgreSQL    │  (Memories/Logs)
        └────────────────┘
                │
        ┌───────▼────────┐
        │ Supermemory.ai │  (Long-term)
        └────────────────┘
```

---

## Next Steps

1. ✅ Choose deployment option (HF Spaces recommended)
2. ✅ Set up infrastructure
3. ✅ Deploy application
4. ✅ Configure monitoring
5. ✅ Implement security
6. ✅ Add content moderation
7. ✅ Go live with beta
8. ✅ Gather feedback
9. ✅ Scale based on demand

---

**Questions?** See README.md or QUICK_START.md
