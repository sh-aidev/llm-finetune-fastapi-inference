from pydantic import BaseModel

class Logger(BaseModel):
    environment: str

class Server(BaseModel):
    host: str
    port: int

class QuantConfig(BaseModel):
    load_in_4bit: bool
    bnb_4bit_use_double_quant: bool
    bnb_4bit_quant_type: str
    bnb_4bit_compute_dtype: str


class LLMModel(BaseModel):
    pretrained_model_name_or_path: str
    use_cache: bool
    attn_implementation: str
    device_map: str
    quantization_config: QuantConfig

class LLMData(BaseModel):
    path: str
    split: str

class PathConfig(BaseModel):
    output_dir: str
    log_dir: str

class LLMTrainer(BaseModel):
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    max_steps: int
    learning_rate: float
    logging_steps: int
    bf16: bool
    tf32: bool
    optim: str
    save_strategy: str
    save_steps: int
    report_to: str

class LLMTraining(BaseModel):
    max_seq_len: int
    packing: bool
    neftune_noise_alpha: float
    trainer: LLMTrainer

class Huggingface(BaseModel):
    push_huggingface: bool
    hf_model_id: str

class Model(BaseModel):
    task_name: str
    max_token_len: int
    top_p: float
    temperature: float
    model: str
    logger: Logger
    server: Server
    llm_model: LLMModel
    llm_data: LLMData
    paths: PathConfig
    llm_training: LLMTraining
    hf: Huggingface

