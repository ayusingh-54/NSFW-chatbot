# 🔥 NSFW Chatbot Fine-Tuning - Combined Dataset Training

## Overview

This project fine-tunes a language model for immersive adult roleplay and sexting using:

1. **HuggingFace Dataset:** `rickRossie/bluemoon_roleplay_chat_data_300k_messages` (~300K roleplay messages)
2. **Custom Datasets:** Your manually curated sexting/roleplay data (~20K+ samples)

---

## 📁 Files Structure

```
NSFW_v0.1/
├── NSFW_FineTune_Combined.ipynb    # Main training notebook (NEW)
├── index.ipynb                      # Original notebook with prompts
├── custom_sexting_dataset.json      # Custom dataset (~13K samples)
├── custom_sexting_dataset_expanded.json  # Expanded dataset
├── lmsys-chat-lewd-filter.prompts.json   # Lewd chat data (~3.5K)
├── merged_dataset.json              # Combined dataset (~8.9K)
├── PROMPTS_SYSTEM_GUIDE.md          # System prompts documentation
└── README_COMBINED_TRAINING.md      # This file
```

---

## ⏱️ Training Time Estimates

### Based on Dataset Size & GPU:

| GPU Type            | 50K Samples  | 100K Samples | 300K Samples |
| ------------------- | ------------ | ------------ | ------------ |
| **T4 (Colab Free)** | ~12-15 hours | ~25-30 hours | ~75-90 hours |
| **V100**            | ~7-9 hours   | ~15-18 hours | ~45-55 hours |
| **A100**            | ~4-5 hours   | ~8-10 hours  | ~25-30 hours |
| **RTX 3090**        | ~8-10 hours  | ~18-22 hours | ~55-65 hours |
| **RTX 4090**        | ~5-6 hours   | ~10-12 hours | ~30-36 hours |

### Calculation Formula:

```
Training Time = (num_samples / effective_batch_size) × epochs × sec_per_step / 3600

Where:
- effective_batch_size = batch_size × gradient_accumulation
- sec_per_step varies by GPU (T4: ~3s, A100: ~1s)
```

### Example (Default Config):

```
Samples: 50,000
Epochs: 3
Batch: 4, Gradient Accum: 4 → Effective: 16
Steps per epoch: 50,000 / 16 = 3,125
Total steps: 3,125 × 3 = 9,375

T4 GPU: 9,375 × 3.0 sec = 28,125 sec ≈ 7.8 hours
A100: 9,375 × 1.0 sec = 9,375 sec ≈ 2.6 hours
```

---

## 🚀 Quick Start

### Option 1: Google Colab (Free GPU)

1. Upload `NSFW_FineTune_Combined.ipynb` to Google Colab
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run all cells in order
4. Training will take ~8-15 hours depending on data size

### Option 2: Local with GPU

```bash
# Install requirements
pip install transformers datasets accelerate peft bitsandbytes trl torch

# Open notebook
jupyter notebook NSFW_FineTune_Combined.ipynb
```

### Option 3: Cloud GPU (Recommended for Full Dataset)

- **Lambda Labs**: A100 @ $1.10/hr
- **RunPod**: A100 @ $0.89/hr
- **Vast.ai**: Various GPUs from $0.20/hr

---

## ⚙️ Configuration Options

### Adjust in Cell 3 of notebook:

```python
CONFIG = {
    # Data Settings
    "max_hf_samples": 50000,      # Reduce for faster training

    # Training Settings
    "num_epochs": 3,              # 2-3 recommended
    "batch_size": 4,              # Reduce if OOM error
    "gradient_accumulation": 4,   # Increase for larger effective batch

    # Model
    "base_model": "mistralai/Mistral-7B-Instruct-v0.2",  # or Llama-2
}
```

### Memory Optimization:

| GPU Memory | Recommended Batch Size  |
| ---------- | ----------------------- |
| 8GB        | 1 (with grad_accum: 16) |
| 12GB       | 2 (with grad_accum: 8)  |
| 16GB       | 4 (with grad_accum: 4)  |
| 24GB+      | 8 (with grad_accum: 2)  |

---

## 📊 Dataset Details

### HuggingFace Dataset: `rickRossie/bluemoon_roleplay_chat_data_300k_messages`

- **Size:** ~300,000 roleplay messages
- **Content:** Adult roleplay conversations from BlueMoon forums
- **Format:** Multi-turn conversations

### Custom Datasets:

| File                                   | Samples  | Content                       |
| -------------------------------------- | -------- | ----------------------------- |
| `custom_sexting_dataset.json`          | ~13,000  | Sexting prompt/response pairs |
| `custom_sexting_dataset_expanded.json` | Variable | Extended dataset              |
| `lmsys-chat-lewd-filter.prompts.json`  | ~3,500   | LMSYS lewd chat data          |
| `merged_dataset.json`                  | ~8,900   | Combined roleplay scenarios   |

### Total Training Data:

- **Minimum (fast):** ~25,000 samples (custom only) → ~4-6 hours
- **Medium:** ~75,000 samples → ~12-18 hours
- **Full:** ~350,000 samples → ~60-90 hours

---

## 🎭 System Prompts Included

### Character Archetypes (9 types):

1. **Dominant** - Commanding, takes control
2. **Submissive Eager** - Enthusiastic, pleasing
3. **Submissive Bratty** - Playful resistance
4. **Seducer (Slow)** - Builds tension gradually
5. **Seducer (Aggressive)** - Direct and intense
6. **Romantic Tender** - Gentle, loving
7. **Romantic Passionate** - Intense emotion
8. **Playful Tease** - Fun, flirty
9. **Dirty Talker** - Explicit narration

### Intensity Levels (5 types):

1. **Soft** - Tender, romantic
2. **Passionate** - Intense but emotional
3. **Raw** - Primal, animalistic
4. **Filthy** - Degradation, taboo
5. **Teasing** - Edging, denial

---

## 💡 Tips for Best Results

### 1. Data Quality > Quantity

- 50K high-quality samples > 300K mixed quality
- Remove very short responses (<50 chars)
- Ensure diverse scenarios

### 2. Training Duration

- **Minimum:** 1 epoch (quick test)
- **Recommended:** 2-3 epochs
- **Maximum:** 4-5 epochs (risk of overfitting)

### 3. Reduce Training Time

```python
# In CONFIG:
"max_hf_samples": 25000,  # Reduce from 50000
"num_epochs": 2,          # Reduce from 3
"batch_size": 8,          # Increase if GPU allows
```

### 4. Monitor Training

- Loss should decrease steadily
- If loss plateaus, training may be done
- Watch for overfitting (eval loss increases)

---

## 🔧 Troubleshooting

### Out of Memory (OOM)

```python
# Reduce batch size
"batch_size": 2,
"gradient_accumulation": 8,  # Keep effective batch same
```

### Slow Training

```python
# Reduce data
"max_hf_samples": 20000,
"num_epochs": 2,
```

### Model Not Learning

```python
# Increase learning rate
"learning_rate": 3e-4,  # Up from 2e-4
```

### Repetitive Outputs

```python
# In generation:
repetition_penalty=1.2,  # Increase from 1.1
temperature=0.9,         # Increase variety
```

---

## 📈 Expected Results

After training, the model should:

✅ Stay in character throughout roleplay
✅ Generate vivid, explicit descriptions
✅ Match user's energy and escalate appropriately
✅ Remember context within conversation
✅ Produce unique responses (not repetitive)
✅ Handle multiple character types
✅ Adjust intensity based on scenario

---

## 🚢 After Training

### Save Locations:

- **LoRA Adapter:** `./nsfw_combined_model/lora_adapter/`
- **Merged Model:** `./nsfw_combined_model/merged_model/`

### To Load Later:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./nsfw_combined_model/merged_model")
tokenizer = AutoTokenizer.from_pretrained("./nsfw_combined_model/merged_model")
```

### Deploy Options:

1. **Local:** Run with `transformers` pipeline
2. **API:** Deploy on RunPod/Modal/Replicate
3. **Gradio:** Create web interface
4. **Ollama:** Convert to GGUF for local inference

---

## ⚠️ Important Notes

1. **GPU Required:** Training without GPU is impractical (would take days/weeks)
2. **Storage:** Merged model needs ~15-20GB disk space
3. **Colab Limits:** Free tier may disconnect during long training
4. **Save Often:** Checkpoints save every 500 steps by default

---

## 📞 Quick Reference

| Task                                   | Time Estimate |
| -------------------------------------- | ------------- |
| Install dependencies                   | 2-3 minutes   |
| Load HuggingFace dataset               | 5-10 minutes  |
| Load base model                        | 5-10 minutes  |
| Training (50K samples, 3 epochs, T4)   | ~8-12 hours   |
| Training (50K samples, 3 epochs, A100) | ~3-4 hours    |
| Save merged model                      | 5-10 minutes  |
| Test generation                        | 1-2 minutes   |

**Total (T4 GPU):** ~10-15 hours
**Total (A100 GPU):** ~4-6 hours

---

## 🎉 Ready to Train!

Open `NSFW_FineTune_Combined.ipynb` and run all cells in order.

Good luck with your training! 🚀
