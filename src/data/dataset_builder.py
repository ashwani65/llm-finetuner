"""
Dataset Builder
Load and format datasets for instruction tuning
"""

from datasets import Dataset, DatasetDict, load_dataset
from typing import List, Dict, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InstructionDatasetBuilder:
    """
    Build instruction-tuning datasets

    Supports multiple formats:
    - SQL generation
    - Code review
    - General Q&A
    """

    def __init__(self, task_type: str = "general"):
        """
        Args:
            task_type: Type of task ('general', 'sql', 'code_review')
        """
        self.task_type = task_type

    def load_from_json(self, file_path: str) -> List[Dict]:
        """Load data from JSON file"""
        logger.info(f"Loading data from {file_path}")

        with open(file_path, 'r') as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data)} examples")
        return data

    def load_from_huggingface(self, dataset_name: str, split: str = "train") -> List[Dict]:
        """Load dataset from HuggingFace Hub"""
        logger.info(f"Loading {dataset_name} from HuggingFace")

        dataset = load_dataset(dataset_name, split=split)
        data = [dict(example) for example in dataset]

        logger.info(f"Loaded {len(data)} examples")
        return data

    def format_for_instruction_tuning(self, examples: List[Dict]) -> List[Dict]:
        """
        Format examples for instruction tuning

        Expected input format (flexible):
        {
            "instruction": "...",  # or "question", "prompt"
            "input": "...",        # optional context
            "output": "..."        # or "response", "answer", "sql"
        }

        Output format:
        {
            "instruction": "...",
            "input": "...",
            "output": "..."
        }
        """
        formatted = []

        for example in examples:
            formatted_example = self._format_single_example(example)
            if formatted_example:
                formatted.append(formatted_example)

        logger.info(f"Formatted {len(formatted)} examples")
        return formatted

    def _format_single_example(self, example: Dict) -> Optional[Dict]:
        """Format a single example based on task type"""

        if self.task_type == "sql":
            return self._format_sql_example(example)
        elif self.task_type == "code_review":
            return self._format_code_review_example(example)
        else:
            return self._format_general_example(example)

    def _format_sql_example(self, example: Dict) -> Dict:
        """
        Format SQL generation example

        Input can have:
        - question + context + sql
        - instruction + input + output
        """
        # Extract fields (flexible field names)
        question = example.get("question") or example.get("instruction", "")
        context = example.get("context") or example.get("schema") or example.get("input", "")
        sql = example.get("sql") or example.get("query") or example.get("output", "")

        if not sql:
            return None

        # Format instruction
        if context:
            instruction = f"Given the following database schema:\n{context}\n\nGenerate a SQL query to answer: {question}"
        else:
            instruction = f"Generate a SQL query to answer: {question}"

        return {
            "instruction": instruction,
            "input": "",
            "output": sql
        }

    def _format_code_review_example(self, example: Dict) -> Dict:
        """Format code review example"""
        code = example.get("code") or example.get("input", "")
        review = example.get("review") or example.get("output", "")

        if not code or not review:
            return None

        instruction = "Review the following code and provide feedback:"

        return {
            "instruction": instruction,
            "input": code,
            "output": review
        }

    def _format_general_example(self, example: Dict) -> Dict:
        """
        Format general instruction-following example

        Handles various field name variations
        """
        # Try to extract instruction
        instruction = (
            example.get("instruction") or
            example.get("question") or
            example.get("prompt") or
            ""
        )

        # Try to extract input/context
        input_text = (
            example.get("input") or
            example.get("context") or
            ""
        )

        # Try to extract output
        output = (
            example.get("output") or
            example.get("response") or
            example.get("answer") or
            example.get("completion") or
            ""
        )

        if not instruction or not output:
            return None

        return {
            "instruction": instruction,
            "input": input_text,
            "output": output
        }

    def create_dataset(
        self,
        data: List[Dict],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> DatasetDict:
        """
        Create train/validation/test splits

        Args:
            data: List of examples
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            seed: Random seed for reproducibility

        Returns:
            DatasetDict with train/validation/test splits
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        logger.info("Creating dataset splits")

        # Format data
        formatted_data = self.format_for_instruction_tuning(data)

        # Create HuggingFace dataset
        dataset = Dataset.from_list(formatted_data)

        # Split: train vs (val + test)
        train_test_split = dataset.train_test_split(
            test_size=(val_ratio + test_ratio),
            seed=seed,
        )

        # Split: val vs test
        val_test_ratio = test_ratio / (val_ratio + test_ratio)
        val_test_split = train_test_split["test"].train_test_split(
            test_size=val_test_ratio,
            seed=seed,
        )

        dataset_dict = DatasetDict({
            "train": train_test_split["train"],
            "validation": val_test_split["train"],
            "test": val_test_split["test"],
        })

        logger.info(f"Dataset splits created:")
        logger.info(f"  Train: {len(dataset_dict['train'])} examples")
        logger.info(f"  Validation: {len(dataset_dict['validation'])} examples")
        logger.info(f"  Test: {len(dataset_dict['test'])} examples")

        return dataset_dict

    def save_dataset(self, dataset_dict: DatasetDict, output_dir: str):
        """Save dataset to disk"""
        logger.info(f"Saving dataset to {output_dir}")
        dataset_dict.save_to_disk(output_dir)
        logger.info("Dataset saved")

    def load_dataset(self, dataset_dir: str) -> DatasetDict:
        """Load dataset from disk"""
        logger.info(f"Loading dataset from {dataset_dir}")
        dataset_dict = DatasetDict.load_from_disk(dataset_dir)
        logger.info("Dataset loaded")
        return dataset_dict


# Convenience functions

def load_and_prepare_dataset(
    file_path: str,
    task_type: str = "general",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> DatasetDict:
    """
    One-line function to load and prepare dataset

    Example:
        dataset = load_and_prepare_dataset("data/sql_examples.json", task_type="sql")
    """
    builder = InstructionDatasetBuilder(task_type=task_type)
    data = builder.load_from_json(file_path)
    return builder.create_dataset(data, train_ratio, val_ratio, test_ratio)


def create_sample_dataset(num_examples: int = 100, task_type: str = "general") -> DatasetDict:
    """
    Create a small sample dataset for testing

    Useful for quick testing without real data
    """
    logger.info(f"Creating sample dataset with {num_examples} examples")

    if task_type == "sql":
        data = [
            {
                "question": f"Find all records where column_{i} > 10",
                "context": f"Table: table_{i} (id, column_{i}, value)",
                "sql": f"SELECT * FROM table_{i} WHERE column_{i} > 10"
            }
            for i in range(num_examples)
        ]
    else:
        data = [
            {
                "instruction": f"Explain concept {i}",
                "input": "",
                "output": f"Concept {i} is an important idea in machine learning that..."
            }
            for i in range(num_examples)
        ]

    builder = InstructionDatasetBuilder(task_type=task_type)
    return builder.create_dataset(data)
