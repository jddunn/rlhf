## RLHF (Reinforcement learning from human feedback) training and usage

This repo comprises a library and modules to allow easy trainig, deploying, and usage of fine-tuned models using Reinforcement Learning from Human Feedback (RLHF), incorporating rewards model alongside Proximal Policy Optimization (PPO) techniques.

By default models are loaded to be quantized with PEFT / LoRA (along with other optimization techniques) to be able to run this more easily on consumer hardware, though this can be turned off. 

RLHF leverages human feedback to guide the learning process, enabling models to improve their predictions over time based on evaluative input. This approach integrates a rewards model to quantitatively assess the quality of model outputs, using these assessments to inform the PPO algorithm's optimization process.

There is a CLI to train and use the models.

Note: This library is not going to be made available via `pip`, as this is just a starting point made open-source, and not meant for production use. 

## Pre-requisites

### Make and activate virtual env

`python -m venv venv`

`source venv/bin/activate`

### Get CUDA drivers

Install CUDA drivers.

Install the appropriate version of PyTorch from the website: https://pytorch.org/get-started/locally/

For example, for Cuda 12.1:

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

### Install the other packages

Install requirements.

`pip install -r requirements.txt`

These libraries were installed on a Windows machine, so they may require different sources / versions. If you run into errors, you'll have to install the right ones manually.

Other packages used: bitsandbytes, TRL, accelerate, transformers

### Make your env file

`cp .env.sample .env`

Get your Hugging Face user access token to download models from their hub: https://huggingface.co/docs/hub/en/security-tokens, and add that to `HUGGING_FACE_HUB_TOKEN` env var.

## RLHF Library usage

### Train a fine-tuned PPO model:

```
from main import NLP
nlp = NLP(
    default_model_name="mistralai/Mistral-7B-v0.1",
    rewards_model_name="facebook/opt-350m",
    models_dir="./nlp/models",
    use_lora=True,
)

ppo_name = nlp.ppo.train_ppo(
    model_name=nlp.model_name,
    ppo_model_name="my_ppo_model",
    dataset_name="ppo_finetune.csv",
    local_only=False,
)

print("Trained PPO model at: " + ppo_name + ".")
```

### Evaluate / predict from a PPO model

```
from main import NLP
ppo_model_name = "my_ppo_model_name"
query = "The movie was really"
nlp = NLP(
    models_dir="./nlp/models",
    load_model=False,
    use_lora=True,
)
nlp.ppo.load_ppo_model(ppo_model_name, local_only=True,     
use_lora=True)
response = nlp.generate_response_from_ppo(query)
print(response)
```

## RLHF CLI usage

### Train a fine-tuned PPO model:

- First arg (after specifying type of CLI, `train` and `ppo`): name of dataset to fine-tune PPO model from (should be a local CSV file inside /data folder or the directory passed as `data_dir` in PPO class, with a column called `query` containing texts / prompts)
- Second arg: name of base model to fine-tune (can be local or from HF hub)
- Third arg: name of the rewards model to use for PPO training (can be local or from HF hub).
- Fourth arg: name of dataset to use for rewards model training (can be local or from HB hub)
- Fith arg: (optional): name of fine-tuned model to save. Defaults to name of base model inside the folder /ppo in /models.

`python cli.py train ppo "ppo_finetune.csv" "meta-llama/Llama-2-7b-chat-hf" "facebook/opt-350m" "Anthropic/hh-rlhf" "ppo-model-finetuned"`

### Evaluate / make prediction from a HF or local model:

- First arg (after specifying type of CLI, `eval` and `model`): Name of model (can be local or from HB hub) to make prediction from
- Second arg: The query to make a prediction from

`python cli.py eval model "meta-llama/Llama-2-7b-chat-hf" "The movie was really"`

### Evaluate / make prediction from a local fine-tuned PPO model (saved inside /models/ppo)

- First arg (after specifying type of CLI, `eval` and `ppo`): Name of model (can be local or from HB hub) to make prediction from
- Second arg: The query to make a prediction from

`python cli.py eval ppo "my-fine-tuned-model-ppo" "The move was really"`

## Linting

`black .`