import torch
from accelerate import Accelerator
from transformers import (
    BitsAndBytesConfig,
)
from peft import LoraConfig

accelerator = Accelerator()


class NLP:
    def __init__(
        self,
        max_tokens_to_return: int = 200,
        adjust_max_tokens_dynamic: bool = True,
        max_tokens_dynamic_ratio: float = 0.75,
    ):
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.quantization_config = BitsAndBytesConfig(
            llm_int8_enable_fp32_cpu_offload=True,
            load_in_8bit=True,
            llm_int8_threshold=5.0,
        )

        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            # task_type="CAUSAL_LM",
            target_modules=[
                # "torch.nn.Linear",
                #  "torch.nn.Embedding"
                #  "torch.nn.Conv2d",
                #  "transformers.pytorch_utils.Conv1D"
            ],
        )
        self.device_map = self.get_device_map_for_value_head()
        self.device = "gpu"
        # Set our max tokens to return in the generate functions to a dynamic
        # amount based on the length of the query. If this is true,
        # max_tokens_to_return will be ignored.
        # Ratio of dynamic max tokens to query tokens length, can be a higher
        # multiplier than 1 to return more tokens than the query length
        self.max_tokens_to_return = max_tokens_to_return
        self.adjust_max_tokens_dynamic = adjust_max_tokens_dynamic
        self.max_tokens_dynamic_ratio = max_tokens_dynamic_ratio

        self.default_generation_kwargs = self.set_default_generation_kwargs(
            max_new_tokens=max_tokens_to_return
        )

        self.setup()
        return

    def setup(self):
        print("Cuda available: ", torch.cuda.is_available())
        if torch.cuda.is_available():
            self.device = "cuda"
        torch.cuda.empty_cache()
        return

    def encode_text(self, text: str, tokenizer, model_device) -> list:
        tensor = tokenizer.encode(text, return_tensors="pt").to(model_device)
        return tensor

    def decode_tensor(
        self,
        tensor,
        tokenizer=None,
    ) -> str:
        text = tokenizer.decode(tensor[0])
        return text

    def calculate_max_tokens_from_query(self, query: str) -> int:
        """ """
        # Return 75% of the length of the tokens in the query, rounded up
        return int(len(query.split(" ")) * self.max_tokens_dynamic_ratio)

    def get_default_generation_kwargs(self):
        return self.default_generation_kwargs

    def set_default_generation_kwargs(
        self,
        query: str = None,
        min_length: int = 1,
        top_k: int = 30,
        top_p: float = 0.9,
        temperature: float = 0.2,
        repetition_penalty: float = 1.02,
        do_sample: bool = True,
        max_new_tokens: int = None,
        adjust_max_tokens_dynamic: bool = None,
    ):
        if max_new_tokens is None:
            max_new_tokens = self.max_tokens_to_return
        if adjust_max_tokens_dynamic is None:
            adjust_max_tokens_dynamic = self.adjust_max_tokens_dynamic
        if adjust_max_tokens_dynamic:
            if query is not None:
                max_new_tokens = self.calculate_max_tokens_from_query(query)
        self.default_generation_kwargs = {
            "min_length": min_length,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
        }
        return self.default_generation_kwargs

    @staticmethod
    def get_device_map_for_value_head():
        device_map = {
            # "model.embed_tokens": "cpu",
            # "model.layers":"cpu",
            # "model.norm":"cpu",
            "model.embed_tokens": 0,
            "model.layers": 0,
            "model.norm": 0,
            "shared": 0,
            "encoder": 0,
            "decoder": 0,
            "transformer.weight": 0,
            "transformer.word_embeddings": 0,
            "transformer.word_embeddings_layernorm": 0,
            "transformer.h": 0,
            "transformer.ln_f": 0,
            "transformer.wte": 0,
            "transformer.wpe": 0,
            "transformer.ln_f": 0,
            # "lm_head": "cpu",
            "lm_head": 0,
            "transformer.h.0": 0,
            "transformer.h.1": 0,
            "transformer.h.2": 0,
            "transformer.h.3": 0,
            "transformer.h.4": 0,
            "transformer.h.5": 0,
            "transformer.h.6": 0,
            "transformer.h.7": 0,
            "transformer.h.8": 0,
            "transformer.h.9": 0,
            "transformer.h.10": 0,
            "transformer.h.11": 0,
            "": Accelerator().local_process_index,
        }
        return device_map
