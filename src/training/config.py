"""
Training Configuration
Dataclasses for LoRA/QLoRA fine-tuning configuration
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List
import yaml
import torch


@dataclass
class LoRAConfig:
    """LoRA-specific configuration"""
    r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA scaling factor
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def to_dict(self):
        return asdict(self)


@dataclass
class QuantizationConfig:
    """Quantization configuration for QLoRA"""
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    def to_dict(self):
        return asdict(self)

    def get_compute_dtype(self):
        """Convert string dtype to torch dtype"""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.bnb_4bit_compute_dtype, torch.float16)


@dataclass
class TrainingConfig:
    """Training hyperparameters and configuration"""

    # Model
    model_name: str = "meta-llama/Llama-2-7b-hf"
    max_seq_length: int = 512
    cache_dir: Optional[str] = None

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Optimization
    optim: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "cosine"

    # Logging and Checkpointing
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

    # Memory Optimization
    fp16: bool = True
    gradient_checkpointing: bool = True

    # Output
    output_dir: str = "./output"
    overwrite_output_dir: bool = True

    # LoRA Configuration
    lora_config: LoRAConfig = field(default_factory=LoRAConfig)

    # Quantization Configuration
    quant_config: QuantizationConfig = field(default_factory=QuantizationConfig)

    # Experiment Tracking
    experiment_name: str = "llm-finetuning"
    run_name: Optional[str] = None
    report_to: str = "none"  # "mlflow", "wandb", "tensorboard", or "none"

    @classmethod
    def from_yaml(cls, path: str):
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Handle nested configs
        if 'lora' in config_dict:
            config_dict['lora_config'] = LoRAConfig(**config_dict.pop('lora'))

        if 'quantization' in config_dict:
            config_dict['quant_config'] = QuantizationConfig(**config_dict.pop('quantization'))

        # Handle training dict
        if 'training' in config_dict:
            training_params = config_dict.pop('training')
            config_dict.update(training_params)

        # Handle model dict
        if 'model' in config_dict:
            model_params = config_dict.pop('model')
            config_dict['model_name'] = model_params.get('name', cls.model_name)
            config_dict['max_seq_length'] = model_params.get('max_seq_length', cls.max_seq_length)
            config_dict['cache_dir'] = model_params.get('cache_dir', cls.cache_dir)

        # Remove any other dicts not in dataclass
        config_dict.pop('tracking', None)
        config_dict.pop('data', None)
        config_dict.pop('inference', None)

        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary"""
        # Handle nested configs
        if 'lora_config' in config_dict and isinstance(config_dict['lora_config'], dict):
            config_dict['lora_config'] = LoRAConfig(**config_dict['lora_config'])

        if 'quant_config' in config_dict and isinstance(config_dict['quant_config'], dict):
            config_dict['quant_config'] = QuantizationConfig(**config_dict['quant_config'])

        return cls(**config_dict)

    def to_yaml(self, path: str):
        """Save configuration to YAML file"""
        config_dict = self.to_dict()
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def to_dict(self):
        """Convert to dictionary"""
        config_dict = asdict(self)
        return config_dict

    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size"""
        return (
            self.per_device_train_batch_size
            * self.gradient_accumulation_steps
            * torch.cuda.device_count() if torch.cuda.is_available() else 1
        )

    def print_config(self):
        """Print configuration summary"""
        print("=" * 60)
        print("Training Configuration")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print(f"Sequence Length: {self.max_seq_length}")
        print(f"\nTraining:")
        print(f"  Epochs: {self.num_train_epochs}")
        print(f"  Batch Size: {self.per_device_train_batch_size}")
        print(f"  Gradient Accumulation: {self.gradient_accumulation_steps}")
        print(f"  Effective Batch Size: {self.get_effective_batch_size()}")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"\nLoRA:")
        print(f"  Rank: {self.lora_config.r}")
        print(f"  Alpha: {self.lora_config.lora_alpha}")
        print(f"  Dropout: {self.lora_config.lora_dropout}")
        print(f"  Target Modules: {', '.join(self.lora_config.target_modules)}")
        print(f"\nQuantization:")
        print(f"  4-bit: {self.quant_config.load_in_4bit}")
        print(f"  Compute dtype: {self.quant_config.bnb_4bit_compute_dtype}")
        print(f"  Quant type: {self.quant_config.bnb_4bit_quant_type}")
        print(f"\nOutput: {self.output_dir}")
        print("=" * 60)
