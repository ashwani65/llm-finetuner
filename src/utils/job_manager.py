"""
Job Manager
Manages async training jobs using threading
"""

import threading
import uuid
import logging
from datetime import datetime
from typing import Dict, Optional
import traceback

from .database import get_db, TrainingJob
from ..training.config import TrainingConfig
from ..training.trainer import LoRAFineTuner
from ..data.dataset_builder import load_and_prepare_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JobManager:
    """
    Manages background training jobs

    Uses threading to run training asynchronously
    """

    def __init__(self):
        self.active_jobs: Dict[str, threading.Thread] = {}
        self.db = get_db()

    def create_job(
        self,
        name: str,
        model_name: str,
        dataset_id: str,
        config: Dict,
    ) -> str:
        """
        Create a new training job

        Returns job_id
        """
        job_id = f"job-{uuid.uuid4().hex[:8]}"

        # Create database entry
        session = self.db.get_session()
        try:
            job = TrainingJob(
                id=job_id,
                name=name,
                model_name=model_name,
                dataset_id=dataset_id,
                config=config,
                status='pending',
            )
            session.add(job)
            session.commit()
            logger.info(f"Created job {job_id}")
        finally:
            session.close()

        return job_id

    def start_job(self, job_id: str):
        """Start a training job in background thread"""
        if job_id in self.active_jobs:
            raise ValueError(f"Job {job_id} is already running")

        # Create and start thread
        thread = threading.Thread(
            target=self._run_training,
            args=(job_id,),
            daemon=True,
        )
        thread.start()
        self.active_jobs[job_id] = thread

        logger.info(f"Started job {job_id}")

    def _run_training(self, job_id: str):
        """
        Run training job (called in background thread)

        This is the actual training execution
        """
        session = self.db.get_session()

        try:
            # Get job from database
            job = session.query(TrainingJob).filter_by(id=job_id).first()
            if not job:
                raise ValueError(f"Job {job_id} not found")

            # Update status
            job.status = 'running'
            job.started_at = datetime.utcnow()
            session.commit()

            logger.info(f"Running training for job {job_id}")

            # Load dataset
            from .database import Dataset
            dataset_record = session.query(Dataset).filter_by(id=job.dataset_id).first()
            if not dataset_record:
                raise ValueError(f"Dataset {job.dataset_id} not found")

            dataset = load_and_prepare_dataset(
                dataset_record.file_path,
                task_type=dataset_record.task_type,
            )

            # Create training config
            config_dict = job.config or {}
            training_config = TrainingConfig(
                model_name=job.model_name,
                num_train_epochs=config_dict.get('num_epochs', 3),
                per_device_train_batch_size=config_dict.get('batch_size', 4),
                learning_rate=config_dict.get('learning_rate', 2e-4),
                output_dir=f"models/{job_id}",
            )

            # Update job config
            job.total_epochs = training_config.num_train_epochs
            job.output_dir = training_config.output_dir
            session.commit()

            # Train model
            fine_tuner = LoRAFineTuner(training_config)
            trainer = fine_tuner.train(dataset, output_dir=training_config.output_dir)

            # Update job with results
            job.status = 'completed'
            job.current_epoch = training_config.num_train_epochs
            job.final_loss = trainer.state.log_history[-1].get('loss', 0) if trainer.state.log_history else None
            job.completed_at = datetime.utcnow()
            session.commit()

            # Create model record
            from .database import Model
            model_id = f"model-{uuid.uuid4().hex[:8]}"
            model = Model(
                id=model_id,
                name=job.name,
                base_model=job.model_name,
                training_job_id=job_id,
                model_path=training_config.output_dir,
                task_type=dataset_record.task_type,
            )
            session.add(model)
            session.commit()

            logger.info(f"Job {job_id} completed successfully")

        except Exception as e:
            logger.error(f"Job {job_id} failed: {str(e)}")
            logger.error(traceback.format_exc())

            # Update job with error
            job = session.query(TrainingJob).filter_by(id=job_id).first()
            if job:
                job.status = 'failed'
                job.error_message = str(e)
                job.completed_at = datetime.utcnow()
                session.commit()

        finally:
            session.close()
            # Remove from active jobs
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]

    def stop_job(self, job_id: str):
        """
        Stop a running job

        Note: Python threading doesn't support forceful termination
        This sets a flag for graceful shutdown
        """
        if job_id not in self.active_jobs:
            raise ValueError(f"Job {job_id} is not running")

        # Update database
        session = self.db.get_session()
        try:
            job = session.query(TrainingJob).filter_by(id=job_id).first()
            if job:
                job.status = 'stopped'
                job.completed_at = datetime.utcnow()
                session.commit()
        finally:
            session.close()

        # Remove from active jobs
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]

        logger.info(f"Stopped job {job_id}")

    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get current job status"""
        session = self.db.get_session()
        try:
            job = session.query(TrainingJob).filter_by(id=job_id).first()
            if job:
                return job.to_dict()
            return None
        finally:
            session.close()

    def list_jobs(self) -> list:
        """List all training jobs"""
        session = self.db.get_session()
        try:
            jobs = session.query(TrainingJob).order_by(TrainingJob.created_at.desc()).all()
            return [job.to_dict() for job in jobs]
        finally:
            session.close()


# Singleton instance
_job_manager = None


def get_job_manager() -> JobManager:
    """Get job manager singleton"""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager
