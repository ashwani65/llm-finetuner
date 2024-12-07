#!/usr/bin/env python3
"""
Evaluation Script
Evaluate fine-tuned models on test data
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import SQLEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_path: str):
    """Load fine-tuned model and tokenizer"""
    logger.info(f"Loading model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    logger.info("Model loaded successfully")
    return model, tokenizer


def generate_predictions(
    model,
    tokenizer,
    test_dataset,
    max_new_tokens: int = 128,
    temperature: float = 0.1,
    batch_size: int = 1
):
    """Generate predictions for test dataset"""
    logger.info("Generating predictions...")

    predictions = []
    model.eval()

    for example in tqdm(test_dataset, desc="Generating"):
        # Format prompt
        instruction = example['instruction']
        input_text = example.get('input', '')

        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract response (after ### Response:)
        if "### Response:" in generated:
            response = generated.split("### Response:")[-1].strip()
        else:
            response = generated.strip()

        predictions.append(response)

    logger.info(f"Generated {len(predictions)} predictions")
    return predictions


def evaluate_predictions(
    predictions,
    references,
    task_type: str = "general"
):
    """Evaluate predictions using appropriate metrics"""
    logger.info(f"Evaluating predictions (task: {task_type})...")

    if task_type == "sql":
        evaluator = SQLEvaluator()
        results = evaluator.evaluate(predictions, references)
    else:
        # General evaluation (BLEU, ROUGE)
        from evaluate import load
        bleu = load("bleu")
        rouge = load("rouge")

        bleu_score = bleu.compute(
            predictions=predictions,
            references=[[ref] for ref in references]
        )

        rouge_scores = rouge.compute(
            predictions=predictions,
            references=references
        )

        results = {
            "bleu": bleu_score["bleu"],
            "rouge": rouge_scores
        }

    return results


def print_results(results, task_type: str = "general"):
    """Print evaluation results"""
    logger.info("="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)

    if task_type == "sql":
        logger.info(f"Exact Match: {results['exact_match']:.4f}")
        logger.info(f"BLEU Score: {results['bleu']:.4f}")
        logger.info("\nComponent Match:")
        for comp, score in results['component_match'].items():
            logger.info(f"  {comp}: {score:.4f}")
        logger.info("\nROUGE Scores:")
        for key, value in results['rouge'].items():
            logger.info(f"  {key}: {value:.4f}")
    else:
        logger.info(f"BLEU Score: {results['bleu']:.4f}")
        logger.info("\nROUGE Scores:")
        for key, value in results['rouge'].items():
            logger.info(f"  {key}: {value:.4f}")

    logger.info("="*60)


def save_results(results, predictions, references, output_path: str):
    """Save evaluation results and predictions"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save results
    results_data = {
        "metrics": results,
        "predictions": predictions[:10],  # Save first 10 for inspection
        "references": references[:10],
        "total_examples": len(predictions)
    }

    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to fine-tuned model")
    parser.add_argument("--test_data", type=str, required=True,
                       help="Path to test data (JSON)")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                       help="Output file for results")
    parser.add_argument("--task_type", type=str, default="general",
                       choices=["general", "sql", "code_review"],
                       help="Task type for evaluation")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Generation temperature")

    args = parser.parse_args()

    # Check GPU
    if not torch.cuda.is_available():
        logger.warning("No GPU detected! Evaluation will be slow.")

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Load test data
    logger.info(f"Loading test data from {args.test_data}")
    test_dataset = load_dataset("json", data_files=args.test_data)["train"]
    logger.info(f"Loaded {len(test_dataset)} test examples")

    # Generate predictions
    predictions = generate_predictions(
        model,
        tokenizer,
        test_dataset,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )

    # Get references
    references = [example["output"] for example in test_dataset]

    # Evaluate
    results = evaluate_predictions(
        predictions,
        references,
        task_type=args.task_type
    )

    # Print results
    print_results(results, task_type=args.task_type)

    # Save results
    save_results(results, predictions, references, args.output)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
