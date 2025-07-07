from types import SimpleNamespace
from typing import List, Mapping, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm

from saprot.model.saprot.saprot_foldseek_mutation_model import (
    SaprotFoldseekMutationModel,
)
from saprot.utils.foldseek_util import get_struc_seq
from saprot.utils.constants import foldseek_struc_vocab


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

        self.pdb_path: str = self.cfg.pdb_path
        self.config_path: str = self.cfg.config_path
        self.threshold: float = self.cfg.threshold
        self.batch_size: int = self.cfg.batch_size

        self.debug: bool = getattr(self.cfg, "debug", False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        parsed_seqs = get_struc_seq("saprot/bin/foldseek", self.pdb_path, ["A"], plddt_mask=False)[
            "A"
        ]
        self.sequecne, self.foldseek_seq, self.combined_seq = parsed_seqs

        config = {
            "foldseek_path": None,
            "config_path": self.config_path,
            "load_pretrained": True,
        }

        self.model = SaprotFoldseekMutationModel(**config).eval().to(self.device)
        self.tokenizer = self.model.tokenizer

    def _log_likelihood(self, batch: Sequence[str]) -> torch.Tensor:
        """
        Args:
            batch (Sequence[str]): バッチ

        Returns:
            torch.Tensor: 各タンパク質の対数尤度
        """
        lls: List[torch.Tensor] = []

        for sequence in batch:
            combined = "".join(aa + self.foldseek_seq[i] for i, aa in enumerate(sequence))

            masked_sequences, positions = [], []
            for idx in range(len(sequence)):
                tokens = self.tokenizer.tokenize(combined)
                tokens[idx] = "#" + tokens[idx][-1]
                masked_sequences.append(" ".join(tokens))
                positions.append(idx + 1)

            ll = 0.0
            for i in range(0, len(masked_sequences), self.batch_size):
                sub = masked_sequences[i : i + self.batch_size]
                inputs = self.tokenizer.batch_encode_plus(
                    sub, return_tensors="pt", padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                probs = self.model.model(**inputs).logits.softmax(dim=-1)  # (B, L, V)

                for j, pos in enumerate(positions[i : i + self.batch_size]):
                    aa = sequence[pos - 1]
                    st = self.tokenizer.get_vocab()[aa + foldseek_struc_vocab[0]]
                    prob = probs[j, pos, st : st + len(foldseek_struc_vocab)].sum()
                    ll += torch.log(prob + 1e-12)

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
        return [sequence for sequence, sc in zip(sequences, scores) if sc >= self.threshold]
