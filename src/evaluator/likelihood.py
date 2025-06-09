from types import SimpleNamespace
from typing import List, Mapping, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


class Likelihood:
    """
    対数尤度でスクリーニングするクラス
    """

    def __init__(self, cfg: Mapping[str, object]) -> None:
        """
        Args:
            cfg (Mapping[str, object]): 設定
        """

        self.cfg = SimpleNamespace(**cfg)

        self.model_name_or_path: str = self.cfg.model_name_or_path
        self.threshold: float = self.cfg.threshold
        self.batch_size: int = self.cfg.batch_size

        self.debug: bool = getattr(self.cfg, "debug", False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    @torch.no_grad()
    def _log_likelihood(self, batch: Sequence[str]) -> torch.Tensor:
        """
        Args:
            batch (Sequence[str]): バッチ

        Returns:
            torch.Tensor: 各タンパク質の対数尤度
        """
        inputs = self.tokenizer(
            list(batch),
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(self.device)
        input_ids = inputs.input_ids  # (B, L)
        attention_mask = inputs.attention_mask  # (B, L)

        logits = self.model(input_ids).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V)
        targets = input_ids[:, 1:]  # (B, L-1)

        # tokens = list("ACDEFGHIKLMNPQRSTVWYBXZUO")
        # ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # first_tok, last_tok = min(ids), max(ids)
        first_tok, last_tok = 5, 29

        logits = logits[:, :, first_tok : last_tok + 1]  # (B, L-1, 25)
        targets = targets - first_tok  # (B, L-1)
        masks = attention_mask[:, 1:]  # (B, L-1)

        lls = []
        for logit, target, mask in zip(logits, targets, masks):
            ll = -F.cross_entropy(
                logit[mask.bool()], target[mask.bool()], reduction="mean"
            )
            lls.append(ll)

        return torch.stack(lls)

    def score(self, sequences: Sequence[str]) -> List[float]:
        """
        Args:
            sequences (Sequence[str]): タンパク質のリスト

        Returns:
            List[float]: 各タンパク質の尤度
        """
        scores = []
        for i in tqdm(range(0, len(sequences), self.batch_size)):
            batch = sequences[i : i + self.batch_size]
            scores.extend(self._log_likelihood(batch).cpu().tolist())
        return scores

    __call__ = score

    def filter(self, sequences: Sequence[str]) -> List[str]:
        """
        Args:
            sequences (Sequence[str]): タンパク質のリスト

        Returns:
            List[str]: スクリーニングされたタンパク質のリスト
        """
        scores = self.score(sequences)
        wt_score = scores[0]
        scores = [wt_score - sc for sc in scores]  # todo: 妥当性の確認
        return [seq for seq, sc in zip(sequences, scores) if sc >= self.threshold]
