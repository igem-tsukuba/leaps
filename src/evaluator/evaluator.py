from __future__ import annotations

from typing import Iterable, List, Sequence

from src.evaluator.functional import Functional
from src.evaluator.likelihood import Likelihood


class Evaluator:
    """
    タンパク質をスクリーニングするクラス
    """

    def __init__(
        self,
        likelihood: Likelihood,
        functionals: Sequence[Functional],
    ) -> None:
        """
        Args:
            likelihood (Likelihood): 尤度でスクリーニングするクラス
            functional (Functional): 機能でスクリーニングするクラス
        """
        self.likelihood = likelihood
        self.functionals = functionals

    def __call__(self, sequences: Iterable[str]) -> List[str]:
        """
        Args:
            sequences (Iterable[str]): タンパク質のイテラブル

        Returns:
            List[str]: スクリーニングされたタンパク質のリスト
        """
        sequences = list(sequences)
        sequences = self.likelihood.filter(sequences)

        for func in self.functionals:
            sequences = func.filter(sequences)

        return sequences

    filter = __call__
