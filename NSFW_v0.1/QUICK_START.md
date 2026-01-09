# Quick Start Guide

## 5-Minute Setup

### 1. Clone/Setup

```bash
cd NSFW_v0.1
```

### 2. Create .env File

```bash
# Copy template and fill in credentials
cp .env.template .env

# Edit .env with your tokens:
# - HF_TOKEN from https://huggingface.co/settings/tokens
# - SUPERMEMORY_API_KEY from https://supermemory.ai/dashboard
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Jupyter

```bash
jupyter notebook nsfw_chatbot_production_v2.ipynb
```

### 5. Run Cells in Order

- Execute all cells in Sections 1-4 (setup, config, security, memory) → ~2 minutes
- Skip Section 5 (model merging) if not enough RAM
- Section 6-7: Load datasets and models → ~10 minutes
- Section 8: Initialize chat engine → ~1 minute
- Section 9: Launch Gradio UI → Done!

**Total: ~15-20 minutes to fully operational chatbot**

---

## Production Deployment Checklist

- [ ] `.env` file created with credentials
- [ ] GPU with 25GB+ VRAM available
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Datasets available (`custom_sexting_dataset.json`, etc.)
- [ ] Model merging completed (optional) OR using base model
- [ ] Fine-tuning completed (optional) OR using pretrained
- [ ] Gradio UI launched successfully
- [ ] Test conversation with sample prompts
- [ ] Memory system working (check short-term buffer)
- [ ] Rate limiting active (test with 11+ requests)
- [ ] Error handling tested
- [ ] Age verification gate added (for public deployment)
- [ ] Model uploaded to HuggingFace Hub
- [ ] Deployed to HF Spaces or Docker

---

## Common Commands

```bash
# Monitor GPU
watch -n 1 nvidia-smi

# Monitor training
tensorboard --logdir ./logs --port 6006

# Check dataset size
wc -l custom_sexting_dataset*.json

# Git status
git status
git add .
git commit -m "Initial commit"

# Push to hub
huggingface-cli repo create nsfw-roleplay-chatbot
huggingface-cli upload nsfw-roleplay-chatbot .
```

---

## Project Files Summary

| File                               | Purpose                               | Status       |
| ---------------------------------- | ------------------------------------- | ------------ |
| `nsfw_chatbot_production_v2.ipynb` | **Main notebook** - Complete pipeline | ✅ Ready     |
| `README.md`                        | Full documentation                    | ✅ Complete  |
| `FINE_TUNING_GUIDE.md`             | Detailed fine-tuning steps            | ✅ Complete  |
| `requirements.txt`                 | Python dependencies                   | ✅ Ready     |
| `.env.template`                    | Environment variables template        | ✅ Ready     |
| `merge_config.yaml`                | Model merging configuration           | ✅ Generated |
| Datasets (JSON files)              | Training data                         | ✅ Provided  |

---

## Performance Expectations

**On A100 80GB GPU:**

- Model loading: 3-5 seconds
- First inference: 5-8 seconds
- Subsequent inferences: 2-3 seconds
- Memory usage: 24-26 GB (with 4-bit)
- Max batch size: 1 (with 1024 token max_length)

**Optimization suggestions:**

- Reduce `max_new_tokens` from 256 to 128 for faster responses
- Use `top_k=30` instead of `top_k=50` for faster generation
- Batch requests using Gradio queue
- Deploy on multiple GPUs for higher throughput

---

## Next: Advanced Usage

- **Memory Enhancement**: Add more Supermemory.ai features
- **Fine-tuning**: Run `trainer.train()` for better quality
- **Deployment**: Push to HF Spaces for public access
- **Monitoring**: Set up logging and metrics collection
- **Scaling**: Use vLLM or text-generation-webui for multi-user support

---

**Questions?** Refer to `README.md` for full documentation.
