import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.utils.logger import logger
from src.utils.config import Config

class LLMInference:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        logger.debug(f"Initializing LLMInference...")
        logger.debug(f"Loading model...")
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            cfg.llm_config.model,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            load_in_4bit=True,
        )
        logger.debug(f"Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_config.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        logger.debug(f"Model and tokenizer loaded...")
        
    def run(self, input_text: str) -> str:
        logger.debug(f"Running inference...")
        input_ids = self.tokenizer(input_text, return_tensors="pt", truncation=True).input_ids.cuda()
        logger.debug(f"Generating output...")
        outputs = self.tokenizer.decode(
            self.model.generate(input_ids=input_ids, max_new_tokens=self.cfg.llm_config.max_token_len, do_sample=True, top_p=self.cfg.llm_config.top_p, temperature=self.cfg.llm_config.temperature)[0],
            skip_special_tokens=True
        )
        logger.debug(f"Generated output: {outputs}")
        return outputs

    def push_to_huggingface(self) -> None:
        logger.debug(f"Pushing model to Huggingface...")
        merged_model = self.model.merge_and_unload()
        logger.debug(f"Model merged and unloaded...")
        merged_model.push_to_hub(self.cfg.hf_model_id)
        logger.debug(f"Model pushed to Huggingface...")
        self.tokenizer.push_to_hub(self.cfg.hf_model_id)
        logger.debug(f"Tokenizer pushed to Huggingface...")

class LLMInferenceHF:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        logger.debug(f"Initializing LLMInferenceHF...")
        logger.debug(f"Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_config.hf.hf_model_id)

        logger.debug(f"Loading model: {cfg.llm_config.hf.hf_model_id}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.llm_config.hf.hf_model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.pipe = pipeline(
            "text-generation", 
            model=self.model, 
            tokenizer=self.tokenizer, 
        )
        logger.debug(f"Pipeline created...")
    
    def run(self, input_text: str) -> str:
        logger.debug(f"Running inference...")
        out = self.pipe(
            input_text,
            max_new_tokens=self.cfg.llm_config.max_token_len,
        )
        logger.debug(f"Generated output: {out[0]['generated_text']}")
        return out[0]['generated_text']
    
    # def push_to_huggingface(self) -> None:
    #     self.model.push_to_hub(self.cfg.hf_model_id)
    #     logger.debug(f"Model pushed to Huggingface...")
    #     self.tokenizer.push_to_hub(self.cfg.hf_model_id)
    #     logger.debug(f"Tokenizer pushed to Huggingface...")