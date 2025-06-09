from types import SimpleNamespace
from typing import List, Mapping, Sequence

import torch
from tqdm import tqdm


class Functional:
    """
    機能でスクリーニングするクラス
    """

    def __init__(self, cfg: Mapping[str, object], predictor: object) -> None:
        """
        Args:
            cfg (Mapping[str, object]): 設定
            predictor (object): 予測器
        """
        self.cfg = SimpleNamespace(**cfg)

        self.threshold: float = self.cfg.threshold
        self.batch_size: int = self.cfg.batch_size

        self.debug: bool = getattr(self.cfg, "debug", False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.predictor = predictor

    @torch.no_grad()
    def _predict(self, batch: Sequence[str]) -> torch.Tensor:
        """
        Args:
            batch (Sequence[str]): バッチ

        Returns:
            torch.Tensor: 各タンパク質の予測値
        """
        preds = self.predictor.predict(batch)
        return torch.as_tensor(preds, device=self.device, dtype=torch.float32)

    def score(self, sequences: Sequence[str]) -> List[float]:
        """
        Args:
            sequences (Sequence[str]): タンパク質のリスト

        Returns:
                List[float]: 各タンパク質の予測値
        """
        scores = []
        for i in tqdm(range(0, len(sequences), self.batch_size)):
            batch = sequences[i : i + self.batch_size]
            scores.extend(self._predict(batch).cpu().tolist())
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
        return [seq for seq, sc in zip(sequences, scores) if sc >= self.threshold]
