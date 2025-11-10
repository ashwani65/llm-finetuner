"""
Evaluation Metrics
Comprehensive metrics for LLM evaluation
"""

from typing import List, Dict
import evaluate
import re
import logging

try:
    import sqlparse
except ImportError:
    sqlparse = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseEvaluator:
    """Base evaluator with common metrics"""

    def __init__(self):
        logger.info("Loading evaluation metrics...")
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        logger.info("Metrics loaded")

    def exact_match(self, predictions: List[str], references: List[str]) -> float:
        """
        Calculate exact match accuracy

        Returns fraction of predictions that exactly match references
        """
        matches = sum(
            pred.strip().lower() == ref.strip().lower()
            for pred, ref in zip(predictions, references)
        )
        return matches / len(predictions) if predictions else 0.0

    def calculate_bleu(self, predictions: List[str], references: List[str]) -> float:
        """
        Calculate BLEU score

        BLEU measures n-gram overlap between prediction and reference
        Higher is better (0-1 scale)
        """
        # Format references as list of lists (required by evaluate library)
        references_formatted = [[ref] for ref in references]

        result = self.bleu.compute(
            predictions=predictions,
            references=references_formatted,
        )

        return result["bleu"]

    def calculate_rouge(self, predictions: List[str], references: List[str]) -> Dict:
        """
        Calculate ROUGE scores

        ROUGE measures recall-based n-gram overlap
        Returns ROUGE-1, ROUGE-2, ROUGE-L scores
        """
        result = self.rouge.compute(
            predictions=predictions,
            references=references,
        )

        return {
            "rouge1": result["rouge1"],
            "rouge2": result["rouge2"],
            "rougeL": result["rougeL"],
        }

    def evaluate(self, predictions: List[str], references: List[str]) -> Dict:
        """Run all base metrics"""
        logger.info(f"Evaluating {len(predictions)} predictions")

        metrics = {
            "exact_match": self.exact_match(predictions, references),
            "bleu": self.calculate_bleu(predictions, references),
        }

        # Add ROUGE scores
        rouge_scores = self.calculate_rouge(predictions, references)
        metrics.update(rouge_scores)

        return metrics


class SQLEvaluator(BaseEvaluator):
    """
    Evaluator for SQL generation tasks

    Includes SQL-specific metrics like component matching
    """

    def __init__(self):
        super().__init__()

    def _normalize_sql(self, sql: str) -> str:
        """
        Normalize SQL query for comparison

        - Remove extra whitespace
        - Convert to lowercase
        - Format with sqlparse if available
        """
        # Remove extra whitespace
        sql = re.sub(r'\s+', ' ', sql.strip().lower())

        # Format with sqlparse if available
        if sqlparse:
            try:
                sql = sqlparse.format(
                    sql,
                    reindent=True,
                    keyword_case='lower'
                )
                sql = re.sub(r'\s+', ' ', sql.strip())
            except Exception as e:
                logger.warning(f"SQL parsing failed: {e}")

        return sql

    def exact_match(self, predictions: List[str], references: List[str]) -> float:
        """
        Exact match with SQL normalization

        Normalizes both queries before comparison
        """
        matches = sum(
            self._normalize_sql(pred) == self._normalize_sql(ref)
            for pred, ref in zip(predictions, references)
        )
        return matches / len(predictions) if predictions else 0.0

    def component_match(self, predictions: List[str], references: List[str]) -> Dict:
        """
        Calculate component-wise accuracy

        Checks if SQL components (SELECT, FROM, WHERE, etc.) are present
        """
        components = ['select', 'from', 'where', 'group by', 'order by', 'join', 'having']
        scores = {comp: 0 for comp in components}
        component_counts = {comp: 0 for comp in components}

        for pred, ref in zip(predictions, references):
            pred_lower = pred.lower()
            ref_lower = ref.lower()

            for comp in components:
                if comp in ref_lower:
                    component_counts[comp] += 1
                    if comp in pred_lower:
                        scores[comp] += 1

        # Normalize by count of references containing each component
        normalized_scores = {}
        for comp in components:
            if component_counts[comp] > 0:
                normalized_scores[comp] = scores[comp] / component_counts[comp]
            else:
                normalized_scores[comp] = None  # Component not present in any reference

        return normalized_scores

    def evaluate(self, predictions: List[str], references: List[str]) -> Dict:
        """Run all SQL-specific metrics"""
        logger.info(f"Evaluating {len(predictions)} SQL predictions")

        # Base metrics
        metrics = super().evaluate(predictions, references)

        # SQL-specific metrics
        metrics["component_match"] = self.component_match(predictions, references)

        return metrics


class CodeEvaluator(BaseEvaluator):
    """
    Evaluator for code generation/review tasks

    Can be extended with code-specific metrics
    """

    def __init__(self):
        super().__init__()

    def evaluate(self, predictions: List[str], references: List[str]) -> Dict:
        """Run code evaluation metrics"""
        logger.info(f"Evaluating {len(predictions)} code predictions")

        # For now, use base metrics
        # Can add code-specific metrics like:
        # - Syntax validity
        # - Code similarity (AST-based)
        # - Execution correctness

        return super().evaluate(predictions, references)


def get_evaluator(task_type: str = "general") -> BaseEvaluator:
    """
    Factory function to get appropriate evaluator

    Args:
        task_type: Type of task ('general', 'sql', 'code')

    Returns:
        Appropriate evaluator instance
    """
    if task_type == "sql":
        return SQLEvaluator()
    elif task_type in ["code", "code_review"]:
        return CodeEvaluator()
    else:
        return BaseEvaluator()


# Convenience function

def evaluate_model(
    predictions: List[str],
    references: List[str],
    task_type: str = "general",
) -> Dict:
    """
    One-line function to evaluate predictions

    Example:
        results = evaluate_model(predictions, references, task_type="sql")
        print(f"BLEU: {results['bleu']:.3f}")
        print(f"Exact Match: {results['exact_match']:.3f}")
    """
    evaluator = get_evaluator(task_type)
    return evaluator.evaluate(predictions, references)
