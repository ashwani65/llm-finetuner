"""
LoRA/QLoRA Fine-Tuner
Core training logic with PEFT library
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)
from datasets import DatasetDict
from typing import Optional, Dict
import os
import logging

from .config import TrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoRAFineTuner:
    """
    Fine-tune LLMs using LoRA/QLoRA with PEFT library

    This implements parameter-efficient fine-tuning:
    - LoRA: Low-Rank Adaptation of large models
    - QLoRA: Quantized LoRA for memory efficiency
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_config = None

    def load_model_and_tokenizer(self):
        """Load base model with optional quantization"""
        logger.info(f"Loading model: {self.config.model_name}")

        # Quantization configuration (for QLoRA)
        if self.config.quant_config.load_in_4bit:
            logger.info("Using 4-bit quantization (QLoRA)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.config.quant_config.get_compute_dtype(),
                bnb_4bit_quant_type=self.config.quant_config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.quant_config.bnb_4bit_use_double_quant,
            )
        else:
            bnb_config = None
            logger.info("Using full precision (standard LoRA)")

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=self.config.cache_dir,
        )

        # Prepare model for k-bit training (required for QLoRA)
        if bnb_config:
            self.model = prepare_model_for_kbit_training(self.model)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            cache_dir=self.config.cache_dir,
        )

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.tokenizer.padding_side = "right"

        logger.info("Model and tokenizer loaded successfully")

    def setup_lora(self):
        """Configure and apply LoRA adapters to the model"""
        logger.info("Setting up LoRA adapters")

        # Create LoRA configuration
        self.peft_config = LoraConfig(
            r=self.config.lora_config.r,
            lora_alpha=self.config.lora_config.lora_alpha,
            lora_dropout=self.config.lora_config.lora_dropout,
            target_modules=self.config.lora_config.target_modules,
            bias=self.config.lora_config.bias,
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply LoRA to model
        self.model = get_peft_model(self.model, self.peft_config)

        # Print trainable parameters
        self.model.print_trainable_parameters()

        logger.info("LoRA adapters applied")

    def prepare_datasets(self, dataset: DatasetDict) -> DatasetDict:
        """
        Tokenize and prepare datasets for training

        Formats data in instruction-tuning format:
        ### Instruction:
        {instruction}

        ### Input:
        {input}

        ### Response:
        {output}
        """
        logger.info("Tokenizing datasets")

        def tokenize_function(examples):
            texts = []
            for instruction, input_text, output in zip(
                examples.get("instruction", [""] * len(examples.get("output", []))),
                examples.get("input", [""] * len(examples.get("output", []))),
                examples["output"],
            ):
                # Format prompt
                if input_text:
                    text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                else:
                    text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

                texts.append(text)

            # Tokenize
            tokenized = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors="pt",
            )

            # For causal LM, labels are same as input_ids
            tokenized["labels"] = tokenized["input_ids"].clone()

            return tokenized

        # Tokenize all splits
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing",
        )

        logger.info(f"Tokenization complete. Train samples: {len(tokenized_dataset['train'])}")

        return tokenized_dataset

    def train(
        self,
        dataset: DatasetDict,
        output_dir: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Trainer:
        """
        Run the training loop

        Args:
            dataset: DatasetDict with 'train' and optionally 'validation' splits
            output_dir: Override config output directory
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Trained Trainer object
        """
        if output_dir:
            self.config.output_dir = output_dir

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        logger.info("="* 60)
        logger.info("Starting Training")
        logger.info("=" * 60)

        # Load model and setup LoRA
        self.load_model_and_tokenizer()
        self.setup_lora()

        # Prepare datasets
        tokenized_dataset = self.prepare_datasets(dataset)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps if "validation" in tokenized_dataset else None,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            evaluation_strategy=self.config.evaluation_strategy if "validation" in tokenized_dataset else "no",
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end if "validation" in tokenized_dataset else False,
            metric_for_best_model=self.config.metric_for_best_model,
            optim=self.config.optim,
            lr_scheduler_type=self.config.lr_scheduler_type,
            fp16=self.config.fp16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            report_to=self.config.report_to,
            overwrite_output_dir=self.config.overwrite_output_dir,
        )

        # Create Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset.get("validation"),
            callbacks=[TrainingProgressCallback()],
        )

        # Train
        logger.info("Starting training loop...")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Log final metrics
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info(f"Training Loss: {train_result.training_loss:.4f}")
        logger.info(f"Training Time: {train_result.metrics['train_runtime']:.2f}s")
        logger.info(f"Samples/sec: {train_result.metrics['train_samples_per_second']:.2f}")
        logger.info("=" * 60)

        # Save final model
        logger.info(f"Saving model to {self.config.output_dir}")
        trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)

        # Save training config
        self.config.to_yaml(os.path.join(self.config.output_dir, "training_config.yaml"))

        return trainer

    def save_merged_model(self, output_dir: str):
        """
        Save model with LoRA weights merged into base model

        This creates a standalone model without PEFT dependencies
        """
        logger.info(f"Merging LoRA weights and saving to {output_dir}")

        if not isinstance(self.model, PeftModel):
            raise ValueError("Model is not a PEFT model")

        # Merge weights
        merged_model = self.model.merge_and_unload()

        # Save
        os.makedirs(output_dir, exist_ok=True)
        merged_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        logger.info("Merged model saved successfully")


class TrainingProgressCallback(TrainerCallback):
    """Custom callback for logging training progress"""

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging"""
        if logs:
            step = state.global_step
            if "loss" in logs:
                logger.info(f"Step {step}: loss={logs['loss']:.4f}")

    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch"""
        logger.info(f"Epoch {state.epoch} complete")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation"""
        if metrics:
            eval_loss = metrics.get("eval_loss", 0)
            logger.info(f"Evaluation - Loss: {eval_loss:.4f}")
