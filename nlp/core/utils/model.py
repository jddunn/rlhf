import torch
from transformers import (
    Adafactor,
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
)
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftModel
from peft import prepare_model_for_int8_training, get_peft_model
from transformers import pipeline
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer


from .access import Access
from .nlp import NLP

# NLP utils
nlp = NLP()

models_tokens_limits = {"meta-llama/Llama-2-7b-chat-hf": 4096}


class Model:
    def __init__(self, use_lora: bool = False):
        self.use_lora = use_lora
        self.tokens_limit = 4096  # Depends on the model we choose
        return

    def _get_model(
        self, model_name: str, local_only: bool = False, use_lora: bool = None
    ) -> tuple:
        _model_name = model_name
        if local_only:
            _model_name = self.models_dir + "/" + model_name
        else:
            Access.login_hf()
            _model_name = model_name
        if use_lora is None:
            use_lora = self.use_lora

        print(
            "Get model: ",
            _model_name,
            "Local only: ",
            local_only,
            "Using Lora: ",
            use_lora,
        )
        # Change token limit based on model
        if _model_name in models_tokens_limits:
            self.tokens_limit = models_tokens_limits[_model_name]
        else:
            self.tokens_limit = 4096
        tokenizer = AutoTokenizer.from_pretrained(_model_name)
        # model = AutoModelForCausalLM.from_pretrained(self.model_name)
        if not use_lora:
            model = AutoModelForCausalLMWithValueHead.from_pretrained(
                _model_name,
                local_files_only=local_only,
                # device_map="balanced",
                device_map=nlp.device_map,
                quantization_config=nlp.quantization_config,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                # load_in_8bit=True,
                load_in_4bit=True,
                trust_remote_code=True,
                # target_model=PeftModel,
                # is_trainable=True
            )
        else:
            model = AutoModelForCausalLMWithValueHead.from_pretrained(
                _model_name,
                local_files_only=local_only,
                # device_map="balanced",
                device_map=nlp.device_map,
                quantization_config=nlp.quantization_config,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                peft_config=nlp.lora_config,
                # target_model=PeftModel,
                # is_trainable=True
            )
            # model = get_peft_model(model, nlp.lora_config)
        # # model = AutoPeftModelForCausalLM.from_pretrained(
        # #     model_name,
        # #     low_cpu_mem_usage=True,
        # #     torch_dtype=torch.float16,
        # #     load_in_4bit=True,
        # #     is_trainable=True,
        # # )

        model.gradient_checkpointing_enable()
        return [tokenizer, model]

    def load_model(self, model_name: str = None, local_only: bool = False) -> None:
        tokenizer, model = self._get_model(model_name, local_only=local_only)
        # Update default generation args with tokenizer pad id
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer = tokenizer
        model = model
        return [tokenizer, model]

    def get_ppo_optimizer(self, model):
        # Update optimizer func for PPO
        optimizer = Adafactor(
            filter(lambda p: p.requires_grad, model.parameters()),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=1.41e-5,
        )
        return optimizer
