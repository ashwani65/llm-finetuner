# Setup Guide

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (for training)
- 16GB+ RAM recommended

## Installation

### 1. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Test the pipeline
python test_pipeline.py
```

Expected output: All tests should pass ✅

## GPU Setup

### Check GPU Availability

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Install CUDA (if needed)

- **CUDA 12.1+**: https://developer.nvidia.com/cuda-downloads
- Verify: `nvidia-smi`

## Quick Start

### 1. Prepare Dataset

```bash
# Use sample dataset (already included)
ls data/sample_sql_dataset.json

# Or prepare your own
python scripts/prepare_data.py \
    --input your_data.json \
    --output data/processed/ \
    --task_type sql
```

### 2. Train Model

```bash
# Start training (requires GPU)
python scripts/train.py \
    --config configs/llama_sql_config.yaml \
    --data data/sample_sql_dataset.json \
    --output models/my-model
```

**Note**: Training a 7B model requires:
- GPU: 16GB+ VRAM (with 4-bit quantization)
- Time: ~2-8 hours depending on dataset size
- Disk: ~500MB for LoRA adapters

### 3. Evaluate Model

```bash
python scripts/evaluate.py \
    --model models/my-model \
    --test_data data/processed/test.json \
    --task_type sql
```

### 4. Deploy Model

```bash
# Start inference server
python scripts/deploy.py \
    --model models/my-model \
    --method local \
    --port 8000
```

## Frontend Setup

### 1. Install Node.js Dependencies

```bash
cd frontend
npm install
```

### 2. Start Frontend

```bash
npm run dev
```

Frontend will be available at: http://localhost:3000

### 3. Start Backend API

```bash
# In another terminal (with venv activated)
cd ..

# Option 1: Use startup script (recommended)
./start_api.sh

# Option 2: Run directly
python -m src.serving.production_api
```

Backend API will be available at: http://localhost:8000

**API Documentation**: http://localhost:8000/docs (Swagger UI)

## Production API Features

The production API (`src/serving/production_api.py`) connects the frontend to the real training pipeline:

### Features
- **Real Database**: SQLite database for tracking jobs, datasets, models, and evaluations
- **Async Training**: Background training using threading (non-blocking API)
- **File Upload**: Upload and manage datasets through the API
- **Job Management**: Create, start, stop, and monitor training jobs
- **Model Registry**: Track and manage trained models
- **Evaluation**: Run evaluations and store results
- **GPU Monitoring**: Check GPU status and utilization

### Key Endpoints
- `POST /datasets/upload` - Upload a new dataset
- `POST /training/start` - Start a training job
- `GET /training/{job_id}/status` - Get training status
- `POST /evaluation/run/{model_id}` - Run evaluation on a model
- `GET /models` - List all trained models
- `GET /system/gpu` - Check GPU status

### Database Schema
The API uses SQLite with the following tables:
- `datasets` - Uploaded datasets
- `training_jobs` - Training job history and status
- `models` - Trained model registry
- `evaluations` - Evaluation results
- `deployments` - Model deployment tracking

Database location: `data/llm_finetuner.db`

## Development Workflow

### Run Tests

```bash
# Test configuration
python test_pipeline.py

# Run unit tests (when implemented)
pytest tests/
```

### Code Formatting

```bash
# Format code
black src/ scripts/

# Lint
flake8 src/ scripts/
```

## Troubleshooting

### Out of Memory Error

**Solution**: Reduce batch size or sequence length in config:
```yaml
training:
  per_device_train_batch_size: 2  # Reduce from 4
  max_seq_length: 256  # Reduce from 512
```

### CUDA Out of Memory

**Solution**:
1. Use 4-bit quantization (QLoRA)
2. Enable gradient checkpointing (already enabled)
3. Use smaller model (e.g., 3B instead of 7B)

### Slow Training

**Solution**:
1. Increase batch size (if you have VRAM)
2. Use gradient accumulation
3. Use multiple GPUs with DDP

### Module Not Found

**Solution**:
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

## Resources

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

## Next Steps

1. ✅ Setup complete - You're ready to train!
2. 📊 Prepare your dataset
3. 🚀 Start training
4. 📈 Evaluate results
5. 🌐 Deploy to production
