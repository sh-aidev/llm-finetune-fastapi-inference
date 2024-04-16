<div align="center">

# LLM Finetuning with FastApi Inference

[![python](https://img.shields.io/badge/-Python_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
![license](https://img.shields.io/badge/License-MIT-green?logo=mit&logoColor=white)

This repository contains the code for finetuning a language model and deploying it using FastApi. We took `mistralai/Mistral-7B-v0.1` model from huggingface and finetuned it on the `c-s-ale/alpaca-gpt4-data` dataset. The model is then deployed using FastApi.

</div>

## 📌 Feature
- [x] Finetuning a language model
- [x] Deploying the model using FastApi
- [x] Added a simple frontend to interact with the FastApi app
- [x] Run finetuned model with vllm backend to get openai like completion

## 📁  Project Structure
The directory structure of new project looks like this:

```
├── LICENSE
├── Makefile
├── README.md
├── __main__.py
├── configs
│   └── config.toml
├── notebooks
│   ├── test.py
│   └── training.ipynb
├── outputs
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── app.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── inference.py
│   │   └── training.py
│   ├── pylogger
│   │   ├── __init__.py
│   │   └── logger.py
│   ├── server
│   │   ├── __init__.py
│   │   └── server.py
│   └── utils
│       ├── __init__.py
│       ├── config.py
│       ├── instruction.py
│       ├── logger.py
│       └── models.py
└── templates
    └── template_alpaca.jinja
```

## 🚀 Getting Started
### Step 1: Clone the repository
```bash
git clone https://github.com/sh-aidev/llm-finetune-fastapi-inference.git

cd llm-finetune-fastapi-inference
```
### Step 2: Install the required dependencies
```bash
python3 -m pip install -r requirements.txt
```

### Step 3: Run the finetuining script
```bash
# Go to configs and change task_name to "train" to train the model in config.toml
python3 __main__.py
```
### Step 4: Run the Inference
```bash
# Go to configs and change task_name to "infer" to run the inference in config.toml
# Change push_huggingface to true after finetuning is complete to push the model to huggingface
python3 __main__.py
```

### Step 5: Run the FastApi server
```bash
# To run the FastApi server change task_name to "server" in configs/config.toml
python3 __main__.py
```
### Step 6: To Run the model vllm backend
```bash
python3 -m vllm.entrypoints.openai.api_server --model "sh-aidev/mistral-7b-v0.1-alpaca-chat" --chat-template ./templates/template_alpaca.jinja --max-model-len 512
```
`NOTE`: Please find the frontend code [here](https://github.com/sh-aidev/openai-chat-clone.git).

## 📜  References
- [Hydra](https://hydra.cc/)
- [FastApi](https://fastapi.tiangolo.com/)
- [Huggingface](https://huggingface.co/)
- [VLLM](https://docs.vllm.ai/en/latest/)
- [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)