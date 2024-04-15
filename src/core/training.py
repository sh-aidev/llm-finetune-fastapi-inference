from src.utils.instruction import format_instruction
from src.utils.logger import logger
from src.utils.config import Config

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model #, AutoPeftModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer
from datetime import datetime
import time

class FinetuningTraining():
    def __init__(self, config: Config) -> None:
        logger.debug(f"Initializing FinetuningTraining...")
        dataset = load_dataset(config.llm_config.llm_data.path, split=config.llm_config.llm_data.split)
        logger.debug(f"Dataset loaded...")

        tokenizer = AutoTokenizer.from_pretrained(config.llm_config.llm_model.pretrained_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        logger.debug(f"Tokenizer loaded...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.llm_config.llm_model.quantization_config.load_in_4bit,
            bnb_4bit_use_double_quant=config.llm_config.llm_model.quantization_config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=config.llm_config.llm_model.quantization_config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=config.llm_config.llm_model.quantization_config.bnb_4bit_compute_dtype
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.llm_config.llm_model.pretrained_model_name_or_path,
            quantization_config=bnb_config,
            use_cache=config.llm_config.llm_model.use_cache,
            attn_implementation=config.llm_config.llm_model.attn_implementation,
            device_map=config.llm_config.llm_model.device_map,
        )
        logger.debug(f"Model loaded...")

        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
            ]
        )

        # prepare model for training
        logger.debug(f"Peft config loaded...")

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        logger.debug(f"Model prepared for kbit training...")

        self.output_dir = Path(config.llm_config.paths.output_dir)
        self.log_dir = Path(config.llm_config.paths.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Output and log directories created...")

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=config.llm_config.llm_training.trainer.per_device_train_batch_size,
            gradient_accumulation_steps=config.llm_config.llm_training.trainer.gradient_accumulation_steps,
            gradient_checkpointing=config.llm_config.llm_training.trainer.gradient_checkpointing,
            max_steps=config.llm_config.llm_training.trainer.max_steps,
            learning_rate=config.llm_config.llm_training.trainer.learning_rate,
            logging_steps=config.llm_config.llm_training.trainer.logging_steps,
            bf16=config.llm_config.llm_training.trainer.bf16,
            tf32=config.llm_config.llm_training.trainer.tf32,
            optim=config.llm_config.llm_training.trainer.optim,
            logging_dir=self.log_dir,
            save_strategy=config.llm_config.llm_training.trainer.save_strategy,
            save_steps=config.llm_config.llm_training.trainer.save_steps,
            report_to=config.llm_config.llm_training.trainer.report_to,
            run_name=f"{config.llm_config.llm_model.pretrained_model_name_or_path}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
        )
        logger.debug(f"Training arguments loaded...")

        self.trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            max_seq_length=config.llm_config.llm_training.max_seq_len,
            tokenizer=tokenizer,
            packing=config.llm_config.llm_training.packing,
            formatting_func=format_instruction,
            args=training_args,
            neftune_noise_alpha=config.llm_config.llm_training.neftune_noise_alpha
        )
        logger.debug(f"Trainer initialized...")

    def run(self):
        logger.debug(f"Running training...")
        self.trainer.train()
        logger.debug(f"Training complete...")

        logger.debug(f"Saving model...")
        self.trainer.save_model()
        logger.debug(f"Model saved...")





