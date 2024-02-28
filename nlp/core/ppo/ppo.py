import os
import torch
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
)
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftModel
from peft import prepare_model_for_int8_training, get_peft_model

import pandas as pd


from .reward import Reward


class PPO:
    def __init__(
        self,
        utils: object = None,
        models_dir: str = "../../models/ppo",
        data_dir: str = "../../data",
        dataset_name: str = "ppo_finetune.csv",
        ppo_model_name: str = None,
        use_lora: bool = False,
        batch_size: int = 1,
        max_shard_size: int = "2000",  # Max shard size to save (in MB),
        reward_model_name: str = "facebook/opt-350m",  # This model has no PyTorch weights which we can't support
        # reward_model_name: str = "weqweasdas/hh_rlhf_rm_open_llama_3b",
        # reward_model_name: str = "facebook/opt-350m",
        reward_dataset_name: str = "Anthropic/hh-rlhf",
    ):
        # PPO model name will be automatically created after training PPO,
        # if it isn't supplied

        self.ppo_config = {"batch_size": batch_size}

        self.ppo_trainer = None

        self.ppo_dataset = ""

        self.ppo_tokenizer = None
        self.ppo_model = None

        self.max_shard_size = max_shard_size

        self.models_dir = models_dir
        self.data_dir = data_dir

        self.dataset_name = dataset_name

        self.utils = utils
        self.utils.files.create_models_dir(self.models_dir)

        self.optimizer = None  # Will be provided after model is loaded

        self.max_tokens_to_return = utils.nlp.max_tokens_to_return
        self.adjust_max_tokens_dynamic = utils.nlp.adjust_max_tokens_dynamic

        self.default_generation_kwargs = utils.nlp.get_default_generation_kwargs()

        self.use_lora = use_lora

        self.reward = Reward(
            model_name=reward_model_name,
            models_dir=os.path.join(self.models_dir, "reward"),
            data_dir=self.data_dir,
            use_lora=self.use_lora,
            dataset_name=reward_dataset_name,
            utils=self.utils,
            # load_from_tf=False,
            # load_from_pt=True,
        )

        return

    def load_ppo_model(
        self,
        model_name: str,
        local_only: bool = False,
        use_lora: bool = None,
        use_value_head: bool = False,
    ) -> None:
        if model_name is None:
            model_name = self.ppo_model_name
        else:
            self.ppo_model_name = model_name
        if local_only:
            model_name = self.models_dir + "/" + model_name
        print("PPO model name to load: ", model_name + ".")

        # model = AutoPeftModelForCausalLM.from_pretrained(name)
        if use_lora is None:
            use_lora = self.use_lora
        if not use_lora:
            print(
                "Loading tokenizer without Lora: "
                + model_name
                + ". "
                + "Local only: "
                + str(local_only)
            )
            self.ppo_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=local_only,
                # padding_side="left"
            )
            self.ppo_tokenizer.pad_token = self.ppo_tokenizer.eos_token
            if not use_value_head:
                print("Loading PPO model without Lora or value head..")
                self.ppo_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    local_files_only=local_only,
                    config=AutoConfig.from_pretrained(model_name),
                    quantization_config=self.utils.nlp.quantization_config,  # attribute will be loaded from saved model
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                    #                                                 # peft_config=lora_config,
                    # is_trainable=True
                )
            else:
                print("Loading PPO model without Lora and with value head..")
                self.ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                    model_name,
                    local_files_only=local_only,
                    config=AutoConfig.from_pretrained(model_name),
                    # quantization_config=self.utils.nlp.quantization_config,  # attribute will be loaded from saved model
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                    #                                                 # peft_config=lora_config,
                    # is_trainable=True
                )
            # .to(self.device)
        else:
            print(
                "Loading tokenizer with Lora "
                + model_name
                + ". "
                + "Local only: "
                + str(local_only)
            )
            # self.ppo_tokenizer = LlamaTokenizer.from_pretrained(
            self.ppo_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=local_only,
                # padding_side="left"
            )
            self.ppo_tokenizer.pad_token = self.ppo_tokenizer.eos_token
            if not use_value_head:
                print("Loading model with Lora and without value head..")
                self.ppo_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    local_files_only=True,
                    # config=AutoConfig.from_pretrained(model_name),
                    # quantization_config=self.utils.nlp.quantization_config,  # attribute will be loaded from saved model
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                    # peft_config=self.lora_config,
                    # is_trainable=True
                )
            else:
                print("Loading model with Lora and with value head..")
                self.ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                    model_name,
                    local_files_only=True,
                    config=AutoConfig.from_pretrained(model_name),
                    # quantization_config=self.utils.nlp.quantization_config,  # attribute will be loaded from saved model
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                    # peft_config=self.lora_config,
                    # is_trainable=True
                )
                self.ppo_model = get_peft_model(self.ppo_model, self.lora_config)
            # .to(self.device)
        print("Loaded model: " + model_name)
        # print("Model:", self.ppo_model)
        return model_name

    def load_dataset(self, dataset_name: str) -> list:
        # Open dataset (CSV file) using csv lib
        dataset = pd.read_csv(self.data_dir + "/" + dataset_name)
        return dataset

    def train_ppo(
        self,
        model_name: str,
        ppo_model_name: str = None,
        dataset_name: str = None,
        reward_model=None,
        generation_kwargs: dict = None,
        max_tokens: int = None,
        local_only: bool = False,
        push_to_hub: bool = False,
    ) -> str:
        print("Initialize PPO trainer..")
        # self.reward.create_trainer()
        self.reward.train()
        # Clone model into reference model
        # tokenizer, model = self.get_model(self.ppo_model_name, local_only=local_only)
        # tokenizer.pad_token = tokenizer.eos_token
        # model_ref = self.get_model(self.ppo_model_name, local_only=local_only)[1]
        tokenizer, model = self.utils.model._get_model(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        self.optimizer = self.utils.model.get_ppo_optimizer(model)

        model_ref = self.utils.model._get_model(model_name, use_lora=True)[1]

        # # model_ref = prepare_model_for_int8_training(model_ref)
        # # model_ref = get_peft_model(model_ref, lora_config)
        config = PPOConfig(**self.ppo_config)

        ppo_reward = None

        if reward_model is None:
            reward_model = self.reward.model

        if dataset_name is None:
            dataset_name = self.dataset_name
        if generation_kwargs is None:
            generation_kwargs = self.default_generation_kwargs

        # 2. initialize trainer
        with torch.cuda.amp.autocast(
            enabled=True, dtype=torch.float16
        ) as autocast, torch.backends.cuda.sdp_kernel(enable_flash=False) as disable:
            # Fix for RuntimeError: Expected is_sm80 || is_sm90 to be true, but got false.
            ppo_config = {"batch_size": 1}
            config = PPOConfig(**ppo_config)
            ppo_trainer = PPOTrainer(
                config,
                model,
                model_ref,
                tokenizer,
                optimizer=self.optimizer,
            )

            dataset = self.load_dataset(dataset_name)
            for query_txt in dataset["query"]:
                query_tensor = tokenizer.encode(query_txt, return_tensors="pt").to(
                    model.device
                )
                if not self.adjust_max_tokens_dynamic:
                    if not max_tokens:
                        max_tokens = self.max_tokens_to_return
                else:
                    if max_tokens is None:
                        max_tokens = self.utils.nlp.calculate_max_tokens_from_query(
                            query_txt
                        )

                generation_kwargs = {
                    "min_length": -1,
                    "top_k": 0.0,
                    "top_p": 1.0,
                    "do_sample": True,
                    "pad_token_id": tokenizer.eos_token_id,
                    "max_new_tokens": max_tokens,
                }
                response_tensor = ppo_trainer.generate(
                    [item for item in query_tensor],
                    return_prompt=False,
                    **generation_kwargs
                )
                response_txt = tokenizer.decode(response_tensor[0])
                print("Response text: ", response_txt)
                ppo_reward = self.reward.get_reward(
                    query_txt, response_txt, model_device=model.pretrained_model.device
                )
                # 6. train model with ppo
                train_stats = ppo_trainer.step(
                    [query_tensor[0]], [response_tensor[0]], ppo_reward
                )
                # print("Train stats: ", train_stats)

        # self.model.eval()

        if not ppo_model_name:
            ppo_model_name = model_name

        if not push_to_hub:
            # peft_model.save_pretrained(lora_adapter, save_adapter=True, save_config=True)
            # self.utils.files.create_models_dir(
            #     os.path.join(self.models_dir, ppo_model_name)
            # )
            tokenizer.save_pretrained(
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        self.models_dir,
                        ppo_model_name,
                    ),
                ),
                safe_serialization=True,
            )
            # model.base_model.save_pretrained(
            #     name,
            #     safe_serialization=True,
            #     max_shard_size="2000MB"
            # )
            # model = model.merge_and_unload()

            model.save_pretrained(
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        self.models_dir,
                        ppo_model_name,
                    )
                ),
                safe_serialization=True,
                max_shard_size=str(self.max_shard_size) + "MB",
            )
        else:
            # Push the model and tokenizer to HF hub
            model.push_to_hub(
                ppo_model_name,
                use_auth_token=True,
                # repo_name="nlp",
            )
        return ppo_model_name
