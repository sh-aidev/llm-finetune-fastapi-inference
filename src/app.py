import os
from src.utils.logger import logger
from src.utils.config import Config
from src.core.training import FinetuningTraining
from src.core.inference import LLMInference, LLMInferenceHF
from src.server.server import LLMServer

class App:
    """
    Main application class to run the FastAPI server. This class will initialize the server and run it.
    """
    def __init__(self) -> None:
        root_config_dir = "configs"
        logger.debug(f"Root config dir: {root_config_dir}")
        self.config = Config(root_config_dir)
        if self.config.llm_config.task_name == "train":
            logger.info("Finetuning mode")
            self.llm = FinetuningTraining(self.config)
        elif self.config.llm_config.task_name == "infer":
            self.llm = LLMInferenceHF(self.config)
        elif self.config.llm_config.task_name == "server":
            self.llm = LLMServer(self.config)

    def run(self):
        if self.config.llm_config.task_name == "infer":
            if self.config.llm_config.hf.push_huggingface == True:
                self.llm.push_to_huggingface()
            INPUT = """
                ### Instruction:
                List 3 historical events related to the following country

                ### Input:
                Canada

                ### Response:
            """
            logger.debug(f"Input: {INPUT}")
            self.llm.run(INPUT)
        else:
            self.llm.run()