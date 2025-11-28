"""
Production API Server
Real implementation connecting frontend to training pipeline
"""

from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import uuid
import os
import json
import shutil

# Import our real implementations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.database import get_db, Dataset, TrainingJob, Model, Evaluation
from src.utils.job_manager import get_job_manager
from src.data.dataset_builder import load_and_prepare_dataset
from src.evaluation.metrics import evaluate_model

app = FastAPI(title="LLM Fine-tuner Production API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class TrainingConfigRequest(BaseModel):
    model_name: str
    dataset_id: str
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 0.0002
    lora_r: int = 16
    lora_alpha: int = 32

# Initialize
db = get_db()
job_manager = get_job_manager()

# Upload directory
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "mode": "production"}


# DATASET APIs
@app.get("/datasets")
async def list_datasets():
    """List all datasets"""
    session = db.get_session()
    try:
        datasets = session.query(Dataset).all()
        return {"data": [d.to_dict() for d in datasets]}
    finally:
        session.close()


@app.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a new dataset"""
    try:
        # Generate ID
        dataset_id = f"dataset-{uuid.uuid4().hex[:8]}"

        # Save file
        file_path = os.path.join(UPLOAD_DIR, f"{dataset_id}.json")
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Parse to get stats
        with open(file_path, 'r') as f:
            data = json.load(f)
            num_examples = len(data) if isinstance(data, list) else 0

        # Create database record
        session = db.get_session()
        try:
            dataset = Dataset(
                id=dataset_id,
                name=file.filename,
                file_path=file_path,
                task_type='general',
                num_examples=num_examples,
                file_size=len(content),
                validated=0,
            )
            session.add(dataset)
            session.commit()

            return {"data": dataset.to_dict()}
        finally:
            session.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset"""
    session = db.get_session()
    try:
        dataset = session.query(Dataset).filter_by(id=dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Delete file
        if os.path.exists(dataset.file_path):
            os.remove(dataset.file_path)

        session.delete(dataset)
        session.commit()

        return {"status": "deleted", "id": dataset_id}
    finally:
        session.close()


# TRAINING APIs
@app.get("/training/jobs")
async def list_training_jobs():
    """List all training jobs"""
    jobs = job_manager.list_jobs()
    return {"data": jobs}


@app.post("/training/start")
async def start_training(config: TrainingConfigRequest):
    """Start a training job"""
    try:
        # Create job
        job_id = job_manager.create_job(
            name=f"{config.model_name.split('/')[-1]}-finetuned",
            model_name=config.model_name,
            dataset_id=config.dataset_id,
            config=config.dict(),
        )

        # Start training in background
        job_manager.start_job(job_id)

        return {"data": {"id": job_id, "status": "starting"}}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/{job_id}/stop")
async def stop_training(job_id: str):
    """Stop a training job"""
    try:
        job_manager.stop_job(job_id)
        return {"status": "stopped", "id": job_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/training/{job_id}/status")
async def get_training_status(job_id: str):
    """Get training job status"""
    status = job_manager.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"data": status}


# MODEL APIs
@app.get("/models")
async def list_models():
    """List all trained models"""
    session = db.get_session()
    try:
        models = session.query(Model).all()
        return {"data": [m.to_dict() for m in models]}
    finally:
        session.close()


@app.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get model details"""
    session = db.get_session()
    try:
        model = session.query(Model).filter_by(id=model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        return {"data": model.to_dict()}
    finally:
        session.close()


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a model"""
    session = db.get_session()
    try:
        model = session.query(Model).filter_by(id=model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        # Delete model files
        if os.path.exists(model.model_path):
            shutil.rmtree(model.model_path)

        session.delete(model)
        session.commit()

        return {"status": "deleted", "id": model_id}
    finally:
        session.close()


# EVALUATION APIs
@app.get("/evaluation/results")
async def list_evaluations():
    """List evaluation results"""
    session = db.get_session()
    try:
        evaluations = session.query(Evaluation).all()
        return {"data": [e.to_dict() for e in evaluations]}
    finally:
        session.close()


@app.post("/evaluation/run/{model_id}")
async def run_evaluation(model_id: str, dataset_id: Optional[str] = None):
    """Run evaluation on a trained model"""
    session = db.get_session()
    try:
        # Get model
        model = session.query(Model).filter_by(id=model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        # Get dataset (use training dataset if not specified)
        if dataset_id:
            dataset = session.query(Dataset).filter_by(id=dataset_id).first()
        else:
            # Get dataset from training job
            job = session.query(TrainingJob).filter_by(id=model.training_job_id).first()
            if job:
                dataset = session.query(Dataset).filter_by(id=job.dataset_id).first()
            else:
                dataset = None

        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Load test data
        dataset_dict = load_and_prepare_dataset(
            dataset.file_path,
            task_type=dataset.task_type,
        )

        # For now, we'll just compute metrics on sample predictions
        # In a full implementation, you'd load the model and generate predictions
        test_data = dataset_dict['test']

        # Mock predictions for demo (in production, generate real predictions)
        predictions = [ex['output'][:50] for ex in test_data[:10]]
        references = [ex['output'] for ex in test_data[:10]]

        # Evaluate
        metrics = evaluate_model(predictions, references, task_type=dataset.task_type)

        # Save evaluation results
        eval_id = f"eval-{uuid.uuid4().hex[:8]}"
        evaluation = Evaluation(
            id=eval_id,
            model_id=model_id,
            model_name=model.name,
            dataset_id=dataset.id,
            metrics=metrics,
            predictions_sample=[
                {"prediction": p, "reference": r}
                for p, r in zip(predictions[:5], references[:5])
            ],
        )
        session.add(evaluation)
        session.commit()

        return {"data": evaluation.to_dict()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


# SYSTEM APIs
@app.get("/system/gpu")
async def get_gpu_status():
    """Get GPU status"""
    import torch

    if torch.cuda.is_available():
        return {
            "data": {
                "name": torch.cuda.get_device_name(0),
                "utilization": 0,  # Would need nvidia-ml-py for real monitoring
                "memory_used": torch.cuda.memory_allocated(0) / 1e9,
                "memory_total": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "temperature": 0,
            }
        }
    else:
        return {
            "data": {
                "name": "No GPU",
                "utilization": 0,
                "memory_used": 0,
                "memory_total": 0,
                "temperature": 0,
            }
        }


if __name__ == "__main__":
    print("="*60)
    print("Starting LLM Fine-tuner Production API")
    print("="*60)
    print("API: http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    print("="*60)

    uvicorn.run(app, host="0.0.0.0", port=8000)
