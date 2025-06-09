from __future__ import annotations

import gc
import random
from pathlib import Path
from types import SimpleNamespace
from typing import List, Mapping, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
from torch import inference_mode
from torch.utils.data import DataLoader
from Bio.Align import substitution_matrices
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, PowerTransformer
from transformers import (
    EsmModel,
    EsmForMaskedLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    logging,
)
from peft import LoraConfig, get_peft_model, TaskType

from src.predictor.regressor import Regressor
from src.predictor.dataset import CustomDataset


logging.set_verbosity_error()


class Predictor:
    """
    回帰モデルの学習と推論をするクラス
    """

    def __init__(self, cfg: Mapping[str, object]) -> None:
        self.cfg = SimpleNamespace(**cfg)

        self.seed: int = self.cfg.seed
        self.debug: bool = self.cfg.debug

        self.target: str = self.cfg.target
        self.num_epochs: int = self.cfg.num_epochs
        self.model_name_or_path: str = self.cfg.model_name_or_path
        self.csv_path: Path = Path(self.cfg.csv_path)
        self.model_path: Path = Path(self.cfg.model_path)
        self.test_size: float = self.cfg.test_size

        self.mutate_per_sample: int = self.cfg.mutate_per_sample
        self.num_mutations: int = self.cfg.num_mutations

        self.destruct_per_samples: int = self.cfg.destruct_per_samples
        self.num_destructions: int = self.cfg.num_destructions

        self.noise_ratio: float = self.cfg.noise_ratio

        self.n_trials: int | None = getattr(self.cfg, "n_trials", None)
        self.timeout: int | None = getattr(self.cfg, "timeout", None)

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = torch.cuda.is_available()
        self.device_type = "cuda" if self.fp16 else "cpu"

        self.best_score = -float("inf")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        self._prepare()

        self.transformer: PowerTransformer
        self.scaler: RobustScaler

        self.study: optuna.Study | None = None
        self.best_params: dict | None = None
        self.state_dict: dict | None = None

    def _prepare(self) -> None:
        df = pd.read_csv(self.csv_path)

        threshold = df[self.target].quantile(0.98)
        df = df[df[self.target] <= threshold]

        self.transformer = PowerTransformer(
            method="box-cox" if (df[self.target] > 0).all() else "yeo-johnson"
        )
        df[self.target] = self.transformer.fit_transform(df[[self.target]]).flatten()

        sequences = df["sequence"].tolist()
        labels = df[self.target].tolist()

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(
            sequences,
            labels,
            test_size=self.test_size,
            random_state=self.seed,
        )

        model = EsmForMaskedLM.from_pretrained(self.model_name_or_path).to(self.device)
        model.eval()

        blosum62 = substitution_matrices.load("BLOSUM62")

        def _conserve(
            aa: str,
            min_score: int | None = None,
            max_score: int | None = None,
        ) -> List[str]:
            """
            Args:
                aa (str): アミノ酸
                min_score (int): 最低スコア
                max_score (int): 最高スコア

            Returns:
                List[str]: 候補のリスト
            """
            index = blosum62.alphabet.index(aa)
            return [
                bb
                for j, bb in enumerate(blosum62.alphabet)
                if bb != aa
                and (min_score is None or blosum62[index, j] >= min_score)
                and (max_score is None or blosum62[index, j] <= max_score)
            ]

        def _destruct(sequence: str) -> Tuple[str, float]:
            """
            Args:
                sequence (str): タンパク質配列
            Returns:
                Tuple[str, float]: 破壊された配列とノイズ
            """
            sequence = list(sequence)
            for pos in random.sample(range(len(sequence)), k=self.num_destructions):
                candidates = _conserve(sequence[pos], max_score=-1)
                if candidates:
                    sequence[pos] = random.choice(candidates)
            noise = random.uniform(-self.noise_ratio, self.noise_ratio)
            return "".join(sequence), noise

        def _mutate(sequence: str) -> Tuple[str, float]:
            """
            Args:
                sequence (str): タンパク質配列

            Returns:
                Tuple[str, float]: 変異された配列とノイズ
            """
            sequence = list(sequence)
            for pos in random.sample(range(len(sequence)), k=self.num_mutations):
                candidates = _conserve(sequence[pos], min_score=1)
                if candidates:
                    sequence[pos] = random.choice(candidates)
            noise = random.uniform(-self.noise_ratio, self.noise_ratio)
            return "".join(sequence), noise

        X_train, y_train = [], []
        with torch.no_grad():
            results = []
            for seq in self.X_train[:]:
                token_ids = self.tokenizer.encode(seq, return_tensors="pt")
                scores = []
                for i in range(1, token_ids.shape[1] - 1):
                    masked_token_ids = token_ids.clone()
                    masked_token_ids[0, i] = self.tokenizer.mask_token_id
                    output = model(masked_token_ids.to(self.device))
                    logits = output.logits[0, i]
                    scores.append(logits[token_ids[0, i]].item())
                results.append((seq, scores))

            results.sort(key=lambda x: np.mean(x[1]), reverse=True)
            for seq, scores in results[: self.destruct_per_samples]:
                indices = np.argsort(scores)[-self.num_destructions :]
                tmp = list(seq)
                for idx in indices:
                    candidates = _conserve(tmp[idx], max_score=-1)
                    if candidates:
                        tmp[idx] = random.choice(candidates)
                sequence, noise = _destruct("".join(tmp))
                X_train.append(sequence)
                y_train.append(0.0 + noise)

        self.X_train.extend(X_train)
        self.y_train.extend(y_train)

        del model
        torch.cuda.empty_cache()
        gc.collect()

        X_train, y_train = [], []
        for x, y in zip(self.X_train, self.y_train):
            if y != 0.0:
                for _ in range(self.mutate_per_sample):
                    sequence, noise = _mutate(x)
                    X_train.append(sequence)
                    y_train.append(y * (1 + noise))

        self.X_train.extend(X_train)
        self.y_train.extend(y_train)

        self.scaler = RobustScaler()
        self.y_train = self.scaler.fit_transform(
            np.array(self.y_train).reshape(-1, 1)
        ).flatten()
        self.y_test = self.scaler.transform(
            np.array(self.y_test).reshape(-1, 1)
        ).flatten()

    def _collate(self, batch: List[Tuple[str, float]]) -> dict[str, torch.Tensor]:
        """
        Args:
            batch (List[Tuple[str, float]]): (タンパク質, ラベル) のタプルのリスト

        Returns:
            dict[str, torch.Tensor]: {"input_ids", "attention_mask", "labels"} の辞書
        """
        sequences, labels = zip(*batch)
        inputs = self.tokenizer(
            list(sequences),
            padding="longest",
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["labels"] = (
            torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(self.device)
        )
        return inputs

    def _objective(self, trial: optuna.trial.Trial) -> float:
        """
        Args:
            trial (optuna.trial.Trial): OptunaのTrial

        Returns:
            float: 決定係数
        """
        lora_r = trial.suggest_categorical("lora_r", [4, 8, 16, 32])
        lora_alpha = trial.suggest_categorical("lora_alpha", [8, 16, 32, 64])
        lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.3, step=0.05)
        lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.25)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

        train_loader = DataLoader(
            CustomDataset(self.X_train, self.y_train),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate,
        )
        val_loader = DataLoader(
            CustomDataset(self.X_test, self.y_test),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate,
        )

        model = EsmModel.from_pretrained(self.model_name_or_path)
        regressor = Regressor(model, model.config.hidden_size).to(self.device)

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"],
        )
        regressor = get_peft_model(regressor, peft_config)

        optimizer = torch.optim.AdamW(
            regressor.parameters(), lr=lr, weight_decay=weight_decay
        )
        num_training_steps = self.num_epochs * len(train_loader)
        warm_steps = int(num_training_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warm_steps,
            num_training_steps=num_training_steps,
        )
        scaler = torch.amp.GradScaler(enabled=self.fp16)

        best_score = -float("inf")
        for epoch in range(self.num_epochs):
            regressor.train()
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=self.device_type, enabled=self.fp16
                ):
                    output = regressor(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = output["loss"]

                if torch.isnan(loss) or torch.isinf(loss):
                    raise optuna.TrialPruned()

                scaler.scale(loss).backward()

                if self.fp16:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(regressor.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            regressor.eval()
            val_pred = []
            val_true = []
            with torch.no_grad(), torch.amp.autocast(
                device_type=self.device_type, enabled=self.fp16
            ):
                for batch in val_loader:
                    output = regressor(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    val_pred.extend(output["logits"].cpu().numpy().flatten())
                    val_true.extend(batch["labels"].cpu().numpy().flatten())

            score = r2_score(
                self.scaler.inverse_transform(np.array(val_true).reshape(-1, 1)),
                self.scaler.inverse_transform(np.array(val_pred).reshape(-1, 1)),
            )

            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            best_score = max(best_score, score)

        if best_score > self.best_score:
            self.best_score = best_score
            torch.save(
                {
                    "state_dict": regressor.state_dict(),
                    "best_params": trial.params,
                },
                self.model_path,
            )

        del regressor, optimizer, scheduler, scaler, model
        torch.cuda.empty_cache()
        gc.collect()
        return best_score

    def train(self) -> None:
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )
        self.study.optimize(
            self._objective, n_trials=self.n_trials, timeout=self.timeout
        )

        self.best_params = self.study.best_params

        print("Best Trial :", self.study.best_trial.number)
        print("Best Score :", self.study.best_value)
        print("Best Params:", self.best_params)

    def load(self) -> None:
        model = torch.load(self.model_path, map_location=self.device)
        self.best_params = model["best_params"]
        self.state_dict = model["state_dict"]

    @inference_mode()
    def predict(self, sequences: List[str]) -> np.ndarray:
        """
        Args:
            sequences (List[str]): 予測するタンパク質のリスト

        Returns:
            np.ndarray: 推論結果の配列
        """
        if self.state_dict is None:
            self.load()

        model = EsmModel.from_pretrained(self.model_name_or_path)
        regressor = Regressor(model, model.config.hidden_size).to(self.device)
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self.best_params["lora_r"],
            lora_alpha=self.best_params["lora_alpha"],
            lora_dropout=self.best_params["lora_dropout"],
            target_modules=["query", "key", "value"],
        )
        regressor = get_peft_model(regressor, peft_config)
        regressor.load_state_dict(self.state_dict)
        regressor.eval()

        inputs = self.tokenizer(
            sequences,
            padding="longest",
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad(), torch.amp.autocast(
            device_type=self.device_type, enabled=self.fp16
        ):
            output = regressor(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        preds = output["logits"].cpu().numpy().flatten()
        preds = self.scaler.inverse_transform(preds.reshape(-1, 1))
        preds = self.transformer.inverse_transform(preds).flatten()

        del regressor, model
        torch.cuda.empty_cache()
        gc.collect()

        return preds
