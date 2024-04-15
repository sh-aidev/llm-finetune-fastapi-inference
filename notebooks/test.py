import torch
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, pipeline
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset
from peft import PeftModel
from trl import SFTTrainer
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed

INSTRUCTION_FORMAT = """
### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}
"""

def format_instruction(sample):
	return INSTRUCTION_FORMAT.format(
        instruction=sample['instruction'],
        input=sample['input'],
        response=sample['output']
    )

alpaca_dataset = load_dataset("c-s-ale/alpaca-gpt4-data", split="train")

model_id = "mistralai/Mistral-7B-v0.1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    use_cache=False,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    # target_modules=[
    #     "q_proj",
    #     "k_proj",
    #     "v_proj",
    #     "o_proj",
    #     "gate_proj",
    #     "up_proj",
    #     "down_proj",
    #     "lm_head",
    # ]
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        # "o_proj",
        # "gate_proj",
    ]
)

# prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
print_trainable_parameters(model)
output_dir = Path("outputs")
logging_dir = Path(output_dir, "logs")
logging_dir.mkdir(parents=True, exist_ok=True)
train_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=6,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    # num_train_epochs=1,
    max_steps=2,
    # max_steps=10,
    # learning_rate=2e-4,
    learning_rate=2.5e-5,
    logging_steps=2,
    bf16=True,
    tf32=True,
    # optim="paged_adamw_8bit",
    optim="paged_adamw_32bit",
    logging_dir=logging_dir,        # Directory for storing logs
    # save_strategy="epoch",
    save_strategy="steps",
    save_steps=50,                # Save checkpoints every 50 steps
    report_to="tensorboard",           # Comment this out if you don't want to use weights & baises
    run_name=f"{model_id}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
)
max_seq_length = 2048 # max sequence length for model and packing of the dataset

trainer = SFTTrainer(
    model=model,
    train_dataset=alpaca_dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=format_instruction,
    args=train_args,
    neftune_noise_alpha=5,
)
trainer.train()