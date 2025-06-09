from __future__ import annotations

from pathlib import Path
import random
from types import SimpleNamespace
from typing import Mapping, List, Optional

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import inference_mode
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    logging,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from datasets import Dataset
from Bio import SeqIO

logging.set_verbosity_error()


class Generator:
    """
    タンパク質を生成するクラス
    """

    def __init__(self, cfg: Mapping[str, object]) -> None:
        """
        Args:
            cfg (Mapping[str, object]): 設定
        """
        self.cfg = SimpleNamespace(**cfg)

        self.seed: int = self.cfg.seed

        fasta_path: Path = Path(self.cfg.fasta_path)
        self.sequences: List[str] = [
            str(rec.seq) for rec in SeqIO.parse(fasta_path, "fasta")
        ]

        test_size: float = self.cfg.test_size
        self.train, self.test = train_test_split(
            self.sequences,
            test_size=test_size,
            random_state=self.seed,
            shuffle=True,
        )

        self.r: int = self.cfg.lora_r
        self.lora_alpha: int = self.cfg.lora_alpha
        self.target_modules: List[str] = self.cfg.target_modules
        self.lora_dropout: float = self.cfg.lora_dropout
        self.model_path: Path = Path(self.cfg.model_path)
        self.model_name_or_path: str = self.cfg.model_name_or_path
        self.weight_decay: float = self.cfg.weight_decay

        self.checkpoint_dir: Path = Path(self.cfg.checkpoint_dir)
        self.per_device_train_batch_size = self.cfg.per_device_train_batch_size
        self.per_device_eval_batch_size = self.cfg.per_device_eval_batch_size
        self.learning_rate: float = self.cfg.learning_rate
        self.num_train_epochs: int = self.cfg.num_train_epochs
        self.gradient_accumulation_steps: int = self.cfg.gradient_accumulation_steps

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float32
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                self.torch_dtype = torch.bfloat16
            else:
                self.torch_dtype = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def train(self) -> None:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
        )
        if isinstance(self.model, PeftModel):
            model = self.model.get_base_model()
        else:
            model = self.model

        self.model = get_peft_model(model, lora_config)

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512,  # todo: ハードコードを避ける
                return_special_tokens_mask=False,
            )

        train_dataset = Dataset.from_dict({"text": self.train}).map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
        )

        eval_dataset = Dataset.from_dict({"text": self.test}).map(
            tokenize_function, batched=True, remove_columns=["text"]
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        training_args = TrainingArguments(
            output_dir=self.checkpoint_dir,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            weight_decay=self.weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            fp16=self.torch_dtype == torch.float16,
            bf16=self.torch_dtype == torch.bfloat16,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            seed=self.seed,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        trainer.train()

        self.model = trainer.model

        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)

        self.model = self.model.to(self.device)
        self.model.eval()

    def load(self) -> None:
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
        )

        self.model = PeftModel.from_pretrained(
            model.to(self.device),
            self.model_path,
            is_trainable=False,
        )

        self.model = self.model.to(self.device)
        self.model.eval()

    @inference_mode()
    def generate(
        self,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        prompt_text: Optional[str] = None,
    ) -> str:
        input_ids = None
        attention_mask = None

        if prompt_text is None or prompt_text == "":
            input_ids = torch.tensor(
                [[self.tokenizer.bos_token_id]],
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = torch.tensor([[1]], dtype=torch.long, device=self.device)
        else:
            inputs = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True,
            ).to(self.device)

            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

        kwargs = {
            "max_length": max_length,
            "min_length": min_length,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "num_return_sequences": 1,
            "pad_token_id": (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        sequence = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return sequence
