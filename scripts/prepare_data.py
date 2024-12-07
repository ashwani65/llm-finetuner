#!/usr/bin/env python3
"""
Data Preparation Script
Prepares and validates datasets for LLM fine-tuning
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
from datasets import Dataset, DatasetDict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_raw_data(input_path: str) -> List[Dict]:
    """Load raw data from JSON file"""
    logger.info(f"Loading data from {input_path}")

    with open(input_path, 'r') as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} examples")
    return data


def format_instruction_data(examples: List[Dict], task_type: str = "general") -> List[Dict]:
    """
    Format examples for instruction tuning

    Expected input format:
    {
        "instruction": "...",
        "input": "...",  # Optional
        "output": "..."
    }
    """
    formatted = []

    for example in examples:
        if task_type == "sql":
            # SQL-specific formatting
            formatted_example = {
                "instruction": f"Given the following database schema:\n{example.get('context', '')}\n\nGenerate a SQL query to answer: {example.get('question', '')}",
                "input": "",
                "output": example.get("sql", example.get("output", ""))
            }
        else:
            # General instruction format
            formatted_example = {
                "instruction": example.get("instruction", ""),
                "input": example.get("input", ""),
                "output": example.get("output", "")
            }

        formatted.append(formatted_example)

    return formatted


def validate_data(data: List[Dict]) -> Dict:
    """Validate dataset quality"""
    df = pd.DataFrame(data)

    # Check for missing values
    missing = df.isnull().sum()

    # Check lengths
    avg_instruction_len = df['instruction'].str.len().mean()
    avg_output_len = df['output'].str.len().mean()

    # Check duplicates
    duplicates = df.duplicated(subset=['instruction', 'output']).sum()

    validation_report = {
        "total_examples": len(df),
        "missing_values": missing.to_dict(),
        "avg_instruction_length": avg_instruction_len,
        "avg_output_length": avg_output_len,
        "duplicates": duplicates
    }

    logger.info("Validation Report:")
    logger.info(f"  Total examples: {validation_report['total_examples']}")
    logger.info(f"  Avg instruction length: {avg_instruction_len:.1f}")
    logger.info(f"  Avg output length: {avg_output_len:.1f}")
    logger.info(f"  Duplicates: {duplicates}")

    return validation_report


def create_splits(
    data: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> DatasetDict:
    """Create train/validation/test splits"""

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    # Create HuggingFace dataset
    dataset = Dataset.from_list(data)

    # First split: train vs (val + test)
    train_test_split = dataset.train_test_split(
        test_size=(val_ratio + test_ratio),
        seed=seed
    )

    # Second split: val vs test
    val_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_test_split = train_test_split["test"].train_test_split(
        test_size=val_test_ratio,
        seed=seed
    )

    dataset_dict = DatasetDict({
        "train": train_test_split["train"],
        "validation": val_test_split["train"],
        "test": val_test_split["test"]
    })

    logger.info("Dataset splits:")
    logger.info(f"  Train: {len(dataset_dict['train'])}")
    logger.info(f"  Validation: {len(dataset_dict['validation'])}")
    logger.info(f"  Test: {len(dataset_dict['test'])}")

    return dataset_dict


def save_dataset(dataset_dict: DatasetDict, output_dir: str):
    """Save dataset splits to JSON files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in dataset_dict.items():
        output_file = output_path / f"{split_name}.json"

        # Convert to list of dicts
        data = [example for example in split_data]

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {split_name} split to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for LLM fine-tuning")
    parser.add_argument("--input", type=str, required=True, help="Input data file (JSON)")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--task_type", type=str, default="general",
                       choices=["general", "sql", "code_review"],
                       help="Task type for formatting")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Load data
    raw_data = load_raw_data(args.input)

    # Format data
    formatted_data = format_instruction_data(raw_data, task_type=args.task_type)

    # Validate data
    validation_report = validate_data(formatted_data)

    # Create splits
    dataset_dict = create_splits(
        formatted_data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    # Save dataset
    save_dataset(dataset_dict, args.output)

    # Save validation report
    report_path = Path(args.output) / "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(validation_report, f, indent=2)

    logger.info(f"Data preparation complete! Saved to {args.output}")


if __name__ == "__main__":
    main()
