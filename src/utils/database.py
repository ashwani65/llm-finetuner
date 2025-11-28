"""
Database Models and Management
SQLite database for tracking training jobs, datasets, and models
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()


class Dataset(Base):
    """Dataset table"""
    __tablename__ = 'datasets'

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    task_type = Column(String, default='general')
    num_examples = Column(Integer, default=0)
    file_size = Column(Integer, default=0)  # bytes
    validated = Column(Integer, default=0)  # boolean (0/1)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'file_path': self.file_path,
            'task_type': self.task_type,
            'examples': self.num_examples,
            'size': f"{self.file_size / 1024:.1f} KB" if self.file_size else "0 KB",
            'validated': bool(self.validated),
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class TrainingJob(Base):
    """Training job table"""
    __tablename__ = 'training_jobs'

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    dataset_id = Column(String)
    config = Column(JSON)  # Training configuration
    status = Column(String, default='pending')  # pending, running, completed, failed
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer, default=3)
    current_loss = Column(Float)
    final_loss = Column(Float)
    accuracy = Column(Float)
    output_dir = Column(String)
    logs = Column(Text)
    error_message = Column(Text)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        duration = None
        if self.started_at and self.completed_at:
            duration = (self.completed_at - self.started_at).total_seconds() / 3600  # hours

        return {
            'id': self.id,
            'name': self.name,
            'model': self.model_name,
            'dataset_id': self.dataset_id,
            'status': self.status,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'current_loss': self.current_loss,
            'final_loss': self.final_loss,
            'accuracy': self.accuracy,
            'output_dir': self.output_dir,
            'duration': duration,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class Model(Base):
    """Trained model table"""
    __tablename__ = 'models'

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    base_model = Column(String, nullable=False)
    training_job_id = Column(String)
    model_path = Column(String, nullable=False)
    model_size = Column(Integer)  # bytes
    task_type = Column(String, default='general')
    metrics = Column(JSON)  # Evaluation metrics
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'base_model': self.base_model,
            'training_job_id': self.training_job_id,
            'model_path': self.model_path,
            'size': f"{self.model_size / (1024**2):.1f} MB" if self.model_size else "0 MB",
            'task_type': self.task_type,
            'metrics': self.metrics,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class Evaluation(Base):
    """Evaluation results table"""
    __tablename__ = 'evaluations'

    id = Column(String, primary_key=True)
    model_id = Column(String, nullable=False)
    model_name = Column(String)
    dataset_id = Column(String)
    metrics = Column(JSON, nullable=False)
    predictions_sample = Column(JSON)  # Store first few predictions
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'model_id': self.model_id,
            'model_name': self.model_name,
            'dataset_id': self.dataset_id,
            'metrics': self.metrics,
            'timestamp': self.created_at.strftime('%Y-%m-%d %H:%M') if self.created_at else None,
        }


class Deployment(Base):
    """Deployment table"""
    __tablename__ = 'deployments'

    id = Column(String, primary_key=True)
    model_id = Column(String, nullable=False)
    model_name = Column(String)
    status = Column(String, default='starting')  # starting, running, stopped, failed
    port = Column(Integer, default=8000)
    endpoint_url = Column(String)
    request_count = Column(Integer, default=0)
    avg_latency = Column(Float, default=0)
    process_id = Column(Integer)  # PID of the server process
    started_at = Column(DateTime)
    stopped_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'model_id': self.model_id,
            'model_name': self.model_name,
            'status': self.status,
            'port': self.port,
            'endpoint_url': self.endpoint_url,
            'request_count': self.request_count,
            'avg_latency': self.avg_latency,
            'started_at': self.started_at.isoformat() if self.started_at else None,
        }


class Database:
    """Database manager"""

    def __init__(self, db_path: str = "data/llm_finetuner.db"):
        """Initialize database connection"""
        # Create data directory if needed
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def get_session(self):
        """Get database session"""
        return self.SessionLocal()

    def close(self):
        """Close database connection"""
        self.engine.dispose()


# Singleton instance
_db_instance = None


def get_db() -> Database:
    """Get database singleton"""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance
