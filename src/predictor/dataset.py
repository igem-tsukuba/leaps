from typing import List, Tuple
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    データセットを保持するクラス

    Args:
        sequences (List[str]): タンパク質のリスト
        labels (List[float]): 目的変数のリスト
    """

    def __init__(self, sequences: List[str], labels: List[float]):
        self.sequences, self.labels = sequences, labels

    def __len__(self) -> int:  # noqa: D401
        return len(self.sequences)

    def __getitem__(self, index: int) -> Tuple[str, float]:
        """
        Args:
            index (int): 取得したいデータのインデックス

        Returns:
            Tuple[str, float]: (タンパク質, ラベル) のタプル
        """
        return self.sequences[index], self.labels[index]
