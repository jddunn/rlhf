# 0. imports
import os
import torch
from transformers import GPT2Tokenizer

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

os.environ["TRANSFORMERS_CACHE"] = "D:\johnny\hugging_face"

from core.ppo.ppo import PPO
from core.utils.utils import Utils


class NLP:
    def __init__(
        self,
        default_model_name: str = "nakodanei/Blue-Orchid-2x7b",
        ppo_model_name: str = None,
        rewads_model_name: str = None,
        models_dir: str = "./models",
        data_dir: str = "./data",
        load_model: bool = True,
        use_lora: bool = False,
        max_tokens_to_return: int = 200,
        adjust_max_tokens_dynamic: bool = True,
        max_tokens_dynamic_ratio: float = 1.5,
    ) -> None:
        self.model_name = default_model_name
        self.ppo_model_name = ppo_model_name

        self.utils = Utils(
            # Set our max tokens to return in the generate functions to a dynamic
            # amount based on the length of the query. If this is true,
            # max_tokens_to_return will be ignored.
            # Ratio of dynamic max tokens to query tokens length, can be a higher
            # multiplier than 1 to return more tokens than the query length
            max_tokens_to_return=max_tokens_to_return,
            adjust_max_tokens_dynamic=adjust_max_tokens_dynamic,
            max_tokens_dynamic_ratio=max_tokens_dynamic_ratio,
            use_lora=use_lora,
        )

        # Are we using Lora for model loading?
        self.utils.model.use_lora = use_lora

        # The vals below will be populated in setup()
        self.bnb_config = self.utils.nlp.bnb_config
        self.lora_config = self.utils.nlp.lora_config
        self.quantization_config = self.utils.nlp.quantization_config
        self.device_map = self.utils.nlp.device_map

        self.models_dir = models_dir

        self.data_dir = data_dir

        self.utils.files.create_data_dir(self.data_dir)

        self.tokenizer = None
        self.model = None

        self.ppo = PPO(utils=self.utils, use_lora=use_lora, data_dir=self.data_dir)
        if load_model:
            self.load_model(self.model_name)
        return

    def load_model(self, model_name: str = None):
        if model_name is None:
            model_name = self.model_name
        print("Loading NLP model.")
        self.tokenizer, self.model = self.utils.model.load_model(model_name=model_name)
        return

    def generate_response(
        self, query: str, generation_kwargs: dict = None, max_tokens: int = None
    ) -> str:
        """ """
        if generation_kwargs is None:
            generation_kwargs = self.utils.nlp.default_generation_kwargs
        if not self.utils.nlp.adjust_max_tokens_dynamic:
            if max_tokens is None:
                max_tokens = self.utils.nlp.max_tokens_to_return
        else:
            if max_tokens is None:
                max_tokens = self.utils.nlp.calculate_max_tokens_from_query(query)
        if max_tokens > self.utils.model.tokens_limit:
            max_tokens = self.utils.model.tokens_limit
        generation_kwargs["max_new_tokens"] = max_tokens
        generation_kwargs["pad_token_id"] = self.tokenizer.eos_token_id
        print("Generating resp for query: ", query, "Max tokens: ", max_tokens)
        inputs = self.utils.nlp.encode_text(
            text=query,
            tokenizer=self.tokenizer,
            model_device=self.model.pretrained_model.device,
        ).to(self.model.pretrained_model.device)
        response_tensor = self.model.generate(
            inputs,
            # return_prompt=False,
            # return_dict_in_generate=True,
            # output_scores=True,
            **generation_kwargs
        )
        print("Resp tensor: ", response_tensor)
        response_text = self.utils.nlp.decode_tensor(
            tensor=response_tensor, tokenizer=self.tokenizer
        )
        print("Resp text: ", response_text)
        return response_text

    def generate_response_from_ppo(
        self, query: str, generation_kwargs: dict = None, max_tokens: int = None
    ) -> str:
        if generation_kwargs is None:
            generation_kwargs = self.utils.nlp.default_generation_kwargs
        if not self.utils.nlp.adjust_max_tokens_dynamic:
            if max_tokens is None:
                max_tokens = self.utils.nlp.max_tokens_to_return
        else:
            if max_tokens is None:
                max_tokens = self.utils.nlp.calculate_max_tokens_from_query(query)
        if max_tokens > self.utils.model.tokens_limit:
            max_tokens = self.utils.model.tokens_limit
        generation_kwargs["max_new_tokens"] = max_tokens
        generation_kwargs["pad_token_id"] = self.ppo.ppo_tokenizer.eos_token_id
        print(
            "Generating resp for query from PPO model: ",
            query,
            "Max tokens: ",
            max_tokens,
        )
        inputs = self.utils.nlp.encode_text(
            text=query,
            tokenizer=self.ppo.ppo_tokenizer,
            model_device=self.ppo.ppo_model.device,
        ).to(self.ppo.ppo_model.device)

        # print("Device: ", self.ppo_model.device)
        response_tensor = self.ppo.ppo_model.generate(
            inputs,
            # fp16=False,
            # return_prompt=False,
            **generation_kwargs
        )
        print("Resp tensor: ", response_tensor)
        response_text = self.utils.nlp.decode_tensor(
            tensor=response_tensor, tokenizer=self.ppo.ppo_tokenizer
        )
        print("Resp text: ", response_text)
        return response_text
