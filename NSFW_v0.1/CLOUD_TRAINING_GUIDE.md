# ☁️ Cloud Training Guide - NSFW Roleplay Chatbot

> **Train your model on cloud GPUs without expensive local hardware**

---

## 📊 Cloud Platform Comparison

| Platform          | Free Tier   | GPU           | VRAM    | Best For      | Cost           |
| ----------------- | ----------- | ------------- | ------- | ------------- | -------------- |
| **Google Colab**  | ✅ Yes      | T4/A100       | 15-40GB | Beginners     | Free-$10/mo    |
| **Kaggle**        | ✅ Yes      | T4x2/P100     | 16-32GB | Free training | Free           |
| **Lightning.ai**  | ✅ Yes      | T4/A10G       | 16-24GB | Easy setup    | Free-$20/mo    |
| **RunPod**        | ❌ No       | RTX 4090/A100 | 24-80GB | Production    | $0.40-$2/hr    |
| **Vast.ai**       | ❌ No       | Various       | 8-80GB  | Budget        | $0.20-$1.50/hr |
| **Lambda Labs**   | ❌ No       | A100/H100     | 40-80GB | Enterprise    | $1.10-$2/hr    |
| **AWS SageMaker** | ❌ Trial    | Various       | 16-80GB | Enterprise    | $1-$4/hr       |
| **Azure ML**      | ❌ Trial    | Various       | 16-80GB | Enterprise    | $1-$4/hr       |
| **Paperspace**    | ✅ Free GPU | M4000-A100    | 8-80GB  | Notebooks     | $0.50-$3/hr    |

---

## 🥇 RECOMMENDED: Google Colab (Easiest)

### Why Colab?

- ✅ **Free tier available** (T4 GPU, 15GB VRAM)
- ✅ **Colab Pro**: $10/mo for A100 access (40GB VRAM)
- ✅ **No setup required** - runs in browser
- ✅ **Google Drive integration** - save models automatically
- ✅ **Pre-installed PyTorch** - faster startup

### Step-by-Step Colab Setup

#### Step 1: Open Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → New Notebook**
3. Click **Runtime → Change runtime type**
4. Select **GPU** (T4 for free, A100 for Pro)
5. Click **Save**

#### Step 2: Check GPU

```python
# Run this first to verify GPU
!nvidia-smi
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

#### Step 3: Mount Google Drive (Save Models)

```python
from google.colab import drive
drive.mount('/content/drive')

# Create project folder
!mkdir -p /content/drive/MyDrive/nsfw_chatbot
%cd /content/drive/MyDrive/nsfw_chatbot
```

#### Step 4: Upload Project Files

```python
# Option A: Upload from local computer
from google.colab import files
uploaded = files.upload()  # Select your JSON dataset files

# Option B: Clone from GitHub (if you have a repo)
# !git clone https://github.com/YOUR_USERNAME/nsfw_chatbot.git

# Option C: Download directly
# !wget https://your-url.com/dataset.json
```

#### Step 5: Install Dependencies

```python
!pip install -q torch==2.0.1 transformers==4.35.2 peft==0.7.1 \
    accelerate==0.24.1 bitsandbytes==0.41.1 datasets==2.14.5 \
    huggingface-hub==0.19.3 gradio==4.11.0
```

#### Step 6: Set HuggingFace Token

```python
import os
from huggingface_hub import login

# Get token from https://huggingface.co/settings/tokens
HF_TOKEN = "hf_your_token_here"  # Replace with your token
login(token=HF_TOKEN)
os.environ["HF_TOKEN"] = HF_TOKEN
```

#### Step 7: Run Training (Complete Code)

```python
import os
import json
import torch
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset, concatenate_datasets

# ═══════════════════════════════════════════════════════════
# CONFIGURATION (Optimized for Colab T4/A100)
# ═══════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    model_name: str = "meta-llama/Llama-2-13b-chat-hf"
    load_in_8bit: bool = True
    max_new_tokens: int = 128
    temperature: float = 0.85

@dataclass
class TrainingConfig:
    output_dir: str = "/content/drive/MyDrive/nsfw_chatbot/adapter"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    max_length: int = 512
    learning_rate: float = 5e-4
    eval_steps: int = 100

model_config = ModelConfig()
training_config = TrainingConfig()

# ═══════════════════════════════════════════════════════════
# LOAD DATASETS
# ═══════════════════════════════════════════════════════════

def load_datasets():
    datasets_list = []

    # Load your JSON files
    json_files = [
        "custom_sexting_dataset.json",
        "custom_sexting_dataset_expanded.json",
        "lmsys-chat-lewd-filter.prompts.json"
    ]

    for file_path in json_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)

            formatted = []
            for entry in data:
                prompt = entry.get('prompt', '').strip()
                completion = entry.get('completion', '').strip()
                if len(prompt) > 20 and len(completion) > 50:
                    formatted.append({
                        "text": f"### Prompt:\n{prompt}\n\n### Response:\n{completion}"
                    })

            if formatted:
                datasets_list.append(Dataset.from_list(formatted))
                print(f"✓ Loaded {len(formatted)} from {file_path}")

    if datasets_list:
        combined = concatenate_datasets(datasets_list)
        split = combined.train_test_split(test_size=0.1, seed=42)
        return split["train"], split["test"]

    raise ValueError("No datasets found!")

train_dataset, eval_dataset = load_datasets()
print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# ═══════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    model_config.model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ═══════════════════════════════════════════════════════════
# TOKENIZE & TRAIN
# ═══════════════════════════════════════════════════════════

def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=training_config.max_length
    )

tokenized_train = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
tokenized_eval = eval_dataset.map(tokenize, batched=True, remove_columns=["text"])

training_args = TrainingArguments(
    output_dir=training_config.output_dir,
    num_train_epochs=training_config.num_train_epochs,
    per_device_train_batch_size=training_config.per_device_train_batch_size,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=training_config.learning_rate,
    warmup_ratio=0.03,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=training_config.eval_steps,
    save_strategy="steps",
    save_steps=200,
    load_best_model_at_end=True,
    bf16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("🚀 Starting training...")
trainer.train()

# Save to Google Drive
trainer.save_model(training_config.output_dir)
tokenizer.save_pretrained(training_config.output_dir)
print(f"✅ Model saved to {training_config.output_dir}")
```

#### Step 8: Test Your Model

```python
# Test generation
from peft import PeftModel

# Reload for inference
base_model = AutoModelForCausalLM.from_pretrained(
    model_config.model_name,
    load_in_8bit=True,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, training_config.output_dir)

def generate(prompt):
    inputs = tokenizer(f"### Prompt:\n{prompt}\n\n### Response:\n", return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.85,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test
print(generate("Hello, how are you today?"))
```

### Colab Tips & Tricks

**⚠️ Session Limits:**

- Free: 12 hours max, may disconnect after 90 min idle
- Pro: 24 hours, priority access
- Pro+: 24 hours, background execution

**💾 Save Progress:**

```python
# Auto-save to Drive every 100 steps
training_args.save_steps = 100
training_args.output_dir = "/content/drive/MyDrive/nsfw_chatbot/checkpoints"
```

**🔄 Resume Training:**

```python
# Resume from checkpoint
trainer.train(resume_from_checkpoint=True)
```

**📊 Monitor GPU:**

```python
# Run in separate cell
!watch -n 1 nvidia-smi
```

---

## 🥈 Kaggle Notebooks (Free T4x2)

### Why Kaggle?

- ✅ **30 hours/week free GPU**
- ✅ **Dual T4 GPUs** (32GB total VRAM)
- ✅ **No disconnection issues**
- ✅ **Direct dataset integration**

### Kaggle Setup

#### Step 1: Create Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **New Notebook**
3. Click **Settings → Accelerator → GPU T4 x2**
4. Enable **Internet** in Settings

#### Step 2: Upload Datasets

1. Click **+ Add Data** in right sidebar
2. Upload your JSON files as a new dataset
3. Access at `/kaggle/input/your-dataset-name/`

#### Step 3: Install & Run

```python
# Same code as Colab, just change paths:
training_config.output_dir = "/kaggle/working/adapter"

# Your datasets are at:
# /kaggle/input/your-dataset-name/custom_sexting_dataset.json
```

#### Step 4: Download Results

```python
# Kaggle auto-saves /kaggle/working/ as output
# Download from notebook output after training
```

---

## 🥉 RunPod (Best Performance)

### Why RunPod?

- ✅ **RTX 4090 for $0.40/hr** (cheapest high-end)
- ✅ **A100 80GB for $1.99/hr**
- ✅ **No time limits**
- ✅ **Persistent storage**
- ✅ **SSH/Jupyter access**

### RunPod Setup

#### Step 1: Create Account

1. Go to [runpod.io](https://www.runpod.io)
2. Sign up and add credits ($10 minimum)

#### Step 2: Deploy GPU Pod

1. Click **Deploy** → **GPU Pods**
2. Select GPU:
   - **RTX 4090** ($0.40/hr) - Best value for this project
   - **A100 40GB** ($1.29/hr) - If you need more VRAM
3. Select Template: **RunPod Pytorch 2.0**
4. Set Volume: **50GB** (for model storage)
5. Click **Deploy**

#### Step 3: Connect

```bash
# Option A: Jupyter (click "Connect" → "Jupyter Lab")

# Option B: SSH
ssh root@your-pod-ip -p 22 -i ~/.ssh/your_key
```

#### Step 4: Setup Project

```bash
# In terminal
cd /workspace
git clone https://github.com/YOUR_USERNAME/nsfw_chatbot.git
cd nsfw_chatbot

# Or upload files via Jupyter
```

#### Step 5: Install & Run

```bash
pip install -r requirements.txt
python train.py  # Or run notebook
```

#### Step 6: Download Model

```bash
# Zip and download via Jupyter file browser
zip -r adapter.zip ./nsfw_adapter_final/
```

### RunPod Cost Calculator

| GPU       | VRAM | $/hr  | 8hr Training | 24hr Training |
| --------- | ---- | ----- | ------------ | ------------- |
| RTX 4090  | 24GB | $0.40 | **$3.20**    | $9.60         |
| RTX A6000 | 48GB | $0.79 | $6.32        | $18.96        |
| A100 40GB | 40GB | $1.29 | $10.32       | $30.96        |
| A100 80GB | 80GB | $1.99 | $15.92       | $47.76        |

**💡 Recommendation:** RTX 4090 for $3.20 total (8hr training)

---

## 🌩️ Vast.ai (Budget Option)

### Why Vast.ai?

- ✅ **Cheapest GPU rentals** (community marketplace)
- ✅ **RTX 3090 from $0.15/hr**
- ✅ **Wide GPU selection**
- ⚠️ Variable reliability (community hosts)

### Vast.ai Setup

#### Step 1: Create Account

1. Go to [vast.ai](https://vast.ai)
2. Sign up and add credits

#### Step 2: Find a Machine

1. Click **Search**
2. Filter:
   - GPU: RTX 4090, RTX 3090, A100
   - VRAM: ≥16GB
   - Disk: ≥50GB
3. Sort by **$/hr**
4. Check reliability score (>95%)

#### Step 3: Rent & Connect

1. Click **Rent**
2. Select **Jupyter** or **SSH**
3. Wait for instance to start
4. Connect via provided link

#### Step 4: Run Training

Same as RunPod - upload files and run notebook

---

## ⚡ Lightning.ai (Easiest Cloud IDE)

### Why Lightning.ai?

- ✅ **22 free GPU hours/month**
- ✅ **VS Code in browser**
- ✅ **One-click setup**
- ✅ **Great for beginners**

### Lightning.ai Setup

#### Step 1: Create Account

1. Go to [lightning.ai](https://lightning.ai)
2. Sign up (GitHub login available)

#### Step 2: Create Studio

1. Click **New Studio**
2. Select **GPU** (T4 free, A10G paid)
3. Choose **Blank** template

#### Step 3: Upload & Run

1. Drag and drop your project files
2. Open terminal: `pip install -r requirements.txt`
3. Open notebook and run cells

---

## 🏢 Enterprise Options

### AWS SageMaker

```python
# SageMaker training script
import sagemaker
from sagemaker.huggingface import HuggingFace

huggingface_estimator = HuggingFace(
    entry_point='train.py',
    source_dir='./src',
    instance_type='ml.g5.2xlarge',  # A10G GPU
    instance_count=1,
    role=sagemaker.get_execution_role(),
    transformers_version='4.35',
    pytorch_version='2.0',
    py_version='py310',
    hyperparameters={
        'epochs': 1,
        'batch_size': 2,
        'learning_rate': 5e-4
    }
)

huggingface_estimator.fit({'training': 's3://your-bucket/data/'})
```

### Azure ML

```python
# Azure ML training
from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="your-subscription",
    resource_group_name="your-rg",
    workspace_name="your-workspace"
)

job = command(
    code="./src",
    command="python train.py",
    environment="AzureML-pytorch-2.0-cuda11.8@latest",
    compute="gpu-cluster",
    instance_count=1
)

ml_client.jobs.create_or_update(job)
```

### Google Cloud Vertex AI

```python
# Vertex AI training
from google.cloud import aiplatform

aiplatform.init(project='your-project', location='us-central1')

job = aiplatform.CustomTrainingJob(
    display_name="nsfw-chatbot-training",
    script_path="train.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-0:latest",
    requirements=["transformers", "peft", "bitsandbytes"]
)

job.run(
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1
)
```

---

## 📱 Mobile/Tablet Training (Yes, Really!)

### Google Colab Mobile

1. Open Chrome on mobile
2. Go to colab.research.google.com
3. Request desktop site
4. Run notebook normally
5. Training runs on Google's servers!

### Paperspace Gradient (Mobile App)

1. Download Paperspace app
2. Create notebook
3. Monitor training on phone

---

## 🔧 Troubleshooting Cloud Issues

### "CUDA out of memory"

```python
# Reduce batch size
training_config.per_device_train_batch_size = 1

# Reduce sequence length
training_config.max_length = 256

# Enable gradient checkpointing (already enabled)
model.gradient_checkpointing_enable()
```

### "Session disconnected" (Colab)

```python
# Keep session alive - run in separate cell
import time
while True:
    time.sleep(60)
    print(".", end="", flush=True)
```

### "Slow training"

```python
# Use bf16 instead of fp16
training_args.bf16 = True
training_args.fp16 = False

# Increase batch size if VRAM allows
training_config.per_device_train_batch_size = 4
```

### "Model too large"

```python
# Switch to smaller model
model_config.model_name = "meta-llama/Llama-2-7b-chat-hf"  # 7B instead of 13B
```

---

## 💰 Cost Comparison Summary

| Platform            | 8hr Training | Best For                |
| ------------------- | ------------ | ----------------------- |
| **Colab Free**      | $0           | Testing, small datasets |
| **Colab Pro**       | $10/mo       | Regular training        |
| **Kaggle**          | $0           | Free serious training   |
| **Lightning.ai**    | $0 (22hr/mo) | Easy setup              |
| **RunPod RTX 4090** | **$3.20**    | Best value production   |
| **Vast.ai**         | $1.50-$5     | Budget option           |
| **AWS/Azure/GCP**   | $10-$30      | Enterprise              |

---

## ✅ Quick Start Checklist

1. [ ] Choose platform (Colab recommended for beginners)
2. [ ] Get HuggingFace token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. [ ] Upload dataset files (JSON)
4. [ ] Select GPU runtime
5. [ ] Run installation cell
6. [ ] Set HF_TOKEN
7. [ ] Run training (~8-10 hours)
8. [ ] Download/save trained model
9. [ ] Test generation

---

## 🎯 Recommended Path

**Beginners:** Google Colab Free → Test with small dataset → Upgrade to Pro

**Budget Users:** Kaggle (free 30hr/week) → Full training for $0

**Production:** RunPod RTX 4090 → $3.20 for complete training

**Enterprise:** AWS/Azure/GCP → Managed infrastructure

---

## 📞 Need Help?

- **Colab Issues:** [colab.research.google.com/notebooks/faq.ipynb](https://colab.research.google.com/notebooks/faq.ipynb)
- **RunPod Discord:** [discord.gg/runpod](https://discord.gg/runpod)
- **HuggingFace Forums:** [discuss.huggingface.co](https://discuss.huggingface.co)

---

**Happy Cloud Training! ☁️🚀**
