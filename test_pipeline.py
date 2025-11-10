#!/usr/bin/env python3
"""
End-to-End Pipeline Test
Test the complete LLM fine-tuning pipeline
"""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.config import TrainingConfig
from src.training.trainer import LoRAFineTuner
from src.data.dataset_builder import load_and_prepare_dataset, create_sample_dataset
from src.evaluation.metrics import evaluate_model

def test_config():
    """Test configuration loading"""
    print("\n" + "="*60)
    print("TEST 1: Configuration")
    print("="*60)

    # Load from YAML
    config = TrainingConfig.from_yaml("configs/llama_sql_config.yaml")
    config.print_config()

    print("✅ Configuration test passed")
    return config


def test_dataset():
    """Test dataset loading and preparation"""
    print("\n" + "="*60)
    print("TEST 2: Dataset Preparation")
    print("="*60)

    # Option 1: Load real dataset
    try:
        dataset = load_and_prepare_dataset(
            "data/sample_sql_dataset.json",
            task_type="sql",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        print(f"✅ Loaded real dataset")
    except FileNotFoundError:
        print("⚠️  Sample dataset not found, creating synthetic data")
        dataset = create_sample_dataset(num_examples=50, task_type="sql")

    print(f"\nDataset sizes:")
    print(f"  Train: {len(dataset['train'])}")
    print(f"  Validation: {len(dataset['validation'])}")
    print(f"  Test: {len(dataset['test'])}")

    # Show sample
    print(f"\nSample example:")
    sample = dataset['train'][0]
    print(f"  Instruction: {sample['instruction'][:100]}...")
    print(f"  Output: {sample['output'][:100]}...")

    print("\n✅ Dataset test passed")
    return dataset


def test_evaluation():
    """Test evaluation metrics"""
    print("\n" + "="*60)
    print("TEST 3: Evaluation Metrics")
    print("="*60)

    # Sample predictions and references
    predictions = [
        "SELECT * FROM users WHERE year = 2024",
        "SELECT category, SUM(revenue) FROM sales GROUP BY category",
    ]

    references = [
        "SELECT * FROM users WHERE YEAR(signup_date) = 2024",
        "SELECT category, SUM(revenue) as total FROM sales GROUP BY category",
    ]

    # Evaluate
    results = evaluate_model(predictions, references, task_type="sql")

    print(f"\nEvaluation Results:")
    print(f"  Exact Match: {results['exact_match']:.3f}")
    print(f"  BLEU Score: {results['bleu']:.3f}")
    print(f"  ROUGE-1: {results['rouge1']:.3f}")
    print(f"  ROUGE-L: {results['rougeL']:.3f}")

    print("\n✅ Evaluation test passed")
    return results


def test_training_setup(config, dataset):
    """Test training setup (without actual training)"""
    print("\n" + "="*60)
    print("TEST 4: Training Setup")
    print("="*60)

    # Check GPU
    if not torch.cuda.is_available():
        print("⚠️  No GPU detected - training will be slow or fail")
        print("   For actual training, you need a CUDA-capable GPU")
    else:
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create trainer
    print("\nInitializing trainer...")
    trainer = LoRAFineTuner(config)

    print("✅ Trainer initialized successfully")
    print("\nTo run actual training:")
    print("  python scripts/train.py --config configs/llama_sql_config.yaml --data data/sample_sql_dataset.json")

    return trainer


def main():
    print("\n" + "="*60)
    print("LLM FINE-TUNING PIPELINE TEST")
    print("="*60)

    try:
        # Test 1: Configuration
        config = test_config()

        # Test 2: Dataset
        dataset = test_dataset()

        # Test 3: Evaluation
        results = test_evaluation()

        # Test 4: Training Setup
        trainer = test_training_setup(config, dataset)

        # Summary
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✅")
        print("="*60)
        print("\nYour LLM fine-tuning pipeline is ready!")
        print("\nNext steps:")
        print("1. Prepare your dataset (or use the sample)")
        print("2. Configure training parameters in configs/")
        print("3. Run training: python scripts/train.py")
        print("4. Evaluate: python scripts/evaluate.py")
        print("5. Deploy: python scripts/deploy.py")

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
