# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Optional

import os
import torch
import gc

import tyro
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from trl import RewardConfig, RewardTrainer, is_xpu_available
from peft import prepare_model_for_int8_training, get_peft_model


tqdm.pandas()


class Reward:
    def __init__(
        self,
        config=None,
        model_name: str = "facebook/opt-350m",
        models_dir: str = "../../models/reward",
        data_dir: str = "../../data",
        dataset_name: str = "Anthropic/hh-rlhf",
        use_lora: bool = True,
        train_split="train",
        eval_split="test",
        load_in_8bit=True,
        load_in_4bit=False,
        trust_remote_code: bool = True,
        # load_from_tf: bool = False,
        # load_from_pt: bool = True,
        utils=None,  # Will be passed from PPO super class
    ) -> None:
        self.config = config
        self.model_name = model_name
        self.models_dir = models_dir

        self.use_lora = use_lora

        self.dataset_name = dataset_name

        self.train_split = train_split
        self.eval_split = eval_split

        # These will be populated when model is loaded
        self.train_dataset = None
        self.eval_dataset = None

        self.tokenizer = None
        self.model = None

        self.trust_remote_code: bool = trust_remote_code
        """Enable `trust_remote_code`"""
        self.reward_config: RewardConfig = RewardConfig(
            output_dir="output",
            per_device_train_batch_size=4,
            num_train_epochs=1,
            gradient_accumulation_steps=16,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=1.41e-5,
            report_to="tensorboard",
            remove_unused_columns=False,
            optim="adamw_torch",
            logging_steps=500,
            evaluation_strategy="no",
            max_length=512,
        )
        self.use_lora: bool = use_lora
        """whether to use peft"""
        self.peft_config: Optional[LoraConfig] = LoraConfig(
            r=16,
            lora_alpha=16,
            bias="none",
            task_type="SEQ_CLS",
            modules_to_save=["scores"],
        )

        self.reward_config.evaluation_strategy = (
            "steps" if self.eval_split != "none" else "no"
        )

        self.quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit
        )
        # Copy the model to each device
        self.device_map = (
            {"": f"xpu:{Accelerator().local_process_index}"}
            if is_xpu_available()
            else {"": Accelerator().local_process_index}
        )

        self.utils = utils
        self.utils.files.create_models_dir(self.models_dir)

        return

    def load_model(
        self, model_name: str = None, local_only: bool = True, use_lora: bool = None
    ) -> list:
        if model_name is None:
            model_name = self.model_name
        else:
            self.model_name = model_name
        if local_only:
            model_name = os.path.join(self.models_dir, model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_only=local_only,
            quantization_config=self.quantization_config,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
            num_labels=1,
        )
        if use_lora is None:
            use_lora = self.use_lora
        # We need logic here to see if the model is just using value head or not
        # to have the PEFT wrapper be supported
        # if use_lora:
        #     model = get_peft_model(model, self.peft_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = model
        self.tokenizer = tokenizer
        return [tokenizer, model]

    def train(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            quantization_config=self.quantization_config,
            device_map=self.device_map,
            num_labels=1,
            trust_remote_code=self.trust_remote_code,
        )
        # We need logic here to see if the model is just using value head or not
        # to have the PEFT wrapper be supported
        # if self.use_lora:
        #     model = get_peft_model(model, self.peft_config)

        # Step 2: Load the dataset and pre-process it
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer = tokenizer
        train_dataset = load_dataset(self.dataset_name, split="train")

        # Preprocess the dataset and filter out examples that are longer than args.max_length
        train_dataset = train_dataset.map(
            self.preprocess_function,
            # fn_kwargs={"tokenizer": tokenizer},
            batched=True,
            num_proc=4,
        )
        train_dataset = train_dataset.filter(
            lambda x: len(x["input_ids_chosen"]) <= self.reward_config.max_length
            and len(x["input_ids_rejected"]) <= self.reward_config.max_length
        )

        eval_dataset = load_dataset(self.dataset_name, split=self.eval_split)

        eval_dataset = eval_dataset.map(
            self.preprocess_function,
            # fn_kwargs={"tokenizer": tokenizer},
            batched=True,
            num_proc=4,
        )
        eval_dataset = eval_dataset.filter(
            lambda x: len(x["input_ids_chosen"]) <= self.reward_config.max_length
            and len(x["input_ids_rejected"]) <= self.reward_config.max_length
        )
        if self.use_lora:
            # Step 5: Define the Trainer
            trainer = RewardTrainer(
                model=model,
                tokenizer=tokenizer,
                args=self.reward_config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                peft_config=self.peft_config,
            )
        else:
            trainer = RewardTrainer(
                model=model,
                tokenizer=tokenizer,
                args=self.reward_config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )
        # print(train_dataset)
        # print(eval_dataset)
        # print(trainer)
        print("Training..")
        gc.collect()
        torch.cuda.empty_cache()
        trainer.train()
        print("Finished training.")

        # self.utils.files.create_models_dir(
        # os.path.join(self.models_dir, self.model_name.replace("/", "_"))
        # )
        print(
            "Saving model..",
            str(
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        self.models_dir,
                        self.model_name,
                    )
                )
            ),
        )
        trainer._save(
            # os.path.join(args.output_dir, "reward"),
            str(
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        self.models_dir,
                        self.model_name,
                    )
                )
            )
            # name=self.model_name + "_" + "reward",
            # model=self.model,
            # tokenizer=self.tokenizer,
        )
        print(
            "Finished saving model at "
            + str(
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        self.models_dir,
                        self.model_name.replace("/", "_"),
                    )
                )
            )
            + "/"
            + self.model_name
            + "_"
            + "reward"
            + "."
        )
        return trainer

    # Tokenize chosen/rejected pairs of inputs
    # Adapt this section to your needs for custom datasets
    def preprocess_function(self, examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tokenized_chosen = self.tokenizer(chosen)
            tokenized_rejected = self.tokenizer(rejected)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(
                tokenized_chosen["attention_mask"]
            )
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(
                tokenized_rejected["attention_mask"]
            )
        print("New examples: ", new_examples)
        return new_examples

    def get_reward(self, query, response, model_device):
        # Tokenize the input pair (query and response)
        tokenized_inputs = self.tokenizer(
            query,
            response,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # Move inputs to the same device as the model
        tokenized_inputs = {k: v.to(model_device) for k, v in tokenized_inputs.items()}

        # Get model predictions
        with torch.no_grad():  # Ensure no gradients are calculated
            outputs = self.model(**tokenized_inputs)

        # Assume the model outputs logits and we're interested in the first value as the reward
        reward = outputs.logits[0].item()  # Convert to Python float for a single value
        return reward


if __name__ == "__main__":
    rt = Reward()
    rt.train()
