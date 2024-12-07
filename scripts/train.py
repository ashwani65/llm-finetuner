#!/usr/bin/env python3
"""
Training Script
Fine-tune LLMs using LoRA/QLoRA
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.config import TrainingConfig
from src.training.trainer import LoRAFineTuner
from datasets import load_dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_gpu():
    """Check GPU availability"""
    if not torch.cuda.is_available():
        logger.warning("No GPU detected! Training will be slow.")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

    logger.info(f"GPU: {gpu_name}")
    logger.info(f"GPU Memory: {gpu_memory:.2f} GB")

    return True


def load_config(config_path: str) -> TrainingConfig:
    """Load training configuration from YAML"""
    logger.info(f"Loading config from {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    config = TrainingConfig.from_dict(config_dict)
    return config


def load_training_data(data_dir: str):
    """Load training datasets"""
    logger.info(f"Loading training data from {data_dir}")

    data_path = Path(data_dir)

    # Load datasets
    train_file = data_path / "train.json"
    val_file = data_path / "validation.json"

    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")

    dataset_files = {
        "train": str(train_file),
        "validation": str(val_file) if val_file.exists() else None
    }

    # Load using datasets library
    dataset = load_dataset("json", data_files={
        k: v for k, v in dataset_files.items() if v is not None
    })

    logger.info(f"Loaded datasets:")
    for split, data in dataset.items():
        logger.info(f"  {split}: {len(data)} examples")

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Train LLM with LoRA/QLoRA")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration YAML file")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to training data directory")
    parser.add_argument("--output", type=str, default="./models/output",
                       help="Output directory for trained model")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume training from checkpoint")

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("LLM Fine-tuning with LoRA/QLoRA")
    logger.info("="*60)

    # Check GPU
    has_gpu = check_gpu()
    if not has_gpu:
        response = input("No GPU detected. Continue anyway? (y/n): ")
        if response.lower() != 'y':
            logger.info("Exiting...")
            return

    # Load configuration
    config = load_config(args.config)

    # Override output directory if provided
    if args.output:
        config.training["output_dir"] = args.output

    # Load training data
    dataset = load_training_data(args.data)

    # Initialize fine-tuner
    logger.info("Initializing fine-tuner...")
    fine_tuner = LoRAFineTuner(config)

    # Train
    logger.info("Starting training...")
    logger.info(f"Model: {config.model['name']}")
    logger.info(f"LoRA rank: {config.lora['r']}")
    logger.info(f"Learning rate: {config.training['learning_rate']}")
    logger.info(f"Epochs: {config.training['num_train_epochs']}")
    logger.info(f"Batch size: {config.training['per_device_train_batch_size']}")

    try:
        trainer = fine_tuner.train(
            dataset=dataset,
            output_dir=args.output,
            resume_from_checkpoint=args.resume
        )

        logger.info("="*60)
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {args.output}")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
