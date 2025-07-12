from __future__ import annotations

import math
from pathlib import Path
import random
from types import SimpleNamespace
from typing import List, Mapping, Dict, Tuple

from Bio.Align import substitution_matrices
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import torch
from transformers import EsmTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers.models.esm.openfold_utils.protein import Protein as OFProtein, to_pdb


class Sampler:
    """
    変異体を生成するクラス
    """

    def __init__(self, cfg: Mapping[str, object]) -> None:
        self.cfg: SimpleNamespace = SimpleNamespace(**cfg)

        fasta_path: Path = Path(self.cfg.fasta_path)
        self.sequences: List[str] = [
            str(rec.seq) for rec in SeqIO.parse(fasta_path, "fasta")
        ]

        self.pdb_path: str = self.cfg.pdb_path
        self.sample_path: Path = Path(self.cfg.sample_path)
        self.num_samples: int = self.cfg.num_samples
        self.mutate_ratio: float = self.cfg.mutate_ratio
        self.alphabet: List[str] = list(self.cfg.alphabet)
        self.temperature: float = self.cfg.temperature

        matrices = substitution_matrices.load("BLOSUM62")
        self.blosum62: Dict[Tuple[str, str], int] = {
            (a, b): matrices[a, b] for a in matrices.alphabet for b in matrices.alphabet
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = (
            EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
            .eval()
            .to(self.device)
        )
        self.tokenizer = EsmTokenizer.from_pretrained("facebook/esmfold_v1")

    def _conserve(self, aa: str, min_score: int = 1) -> List[str]:
        """
        Args:
            aa (str): アミノ酸
            min_score (int): スコア

        Returns:
            List[str]: 候補のリスト
        """
        return [
            bb
            for bb in self.alphabet
            if self.blosum62[(aa, bb)] >= min_score and bb != aa
        ]

    def _mutate(self, sequence: str) -> str:
        """
        Args:
            sequence (str): 変異させるタンパク質
        Returns:
            str: 変異されたタンパク質
        """
        sequence = list(sequence)
        for i, aa in enumerate(sequence):
            if random.random() < self.mutate_ratio:
                candidates = self._conserve(aa, min_score=1)
                if not candidates:
                    continue

                weights = [
                    math.exp(self.blosum62[(aa, cand)] / self.temperature)
                    for cand in candidates
                ]

                sequence[i] = random.choices(candidates, weights, k=1)[0]
        return "".join(sequence)

    def _shuffle(self, parent1: str, parent2: str) -> str:
        """
        Args:
            parent1 (str): 1つ目のタンパク質
            parent2 (str): 2つ目のタンパク質

        Returns:
            str: シャッフルされたタンパク質
        """
        length = len(parent1)

        points = sorted(
            random.sample(range(1, length), 5 - 1)
        )  # todo: ハードコードを避ける
        points = [0] + points + [length]

        sequence = []
        for idx in range(len(points) - 1):
            start, end = points[idx], points[idx + 1]
            parent = parent1 if idx % 2 == 0 else parent2
            sequence.append(parent[start:end])
        return "".join(sequence)

    def _get_pdbstr(self, sequence: str) -> str:
        """
        Args:
            sequence (str): タンパク質の配列

        Returns:
            str: PDB形式の文字列
        """
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        atom_positions = atom14_to_atom37(outputs.positions[-1], outputs)

        protein = OFProtein(
            aatype=outputs.aatype[0].cpu().numpy(),
            atom_positions=atom_positions[0].cpu().numpy(),
            atom_mask=outputs.atom37_atom_exists[0].cpu().numpy(),
            residue_index=outputs.residue_index[0].cpu().numpy() + 1,
            b_factors=outputs.plddt[0].cpu().numpy(),
        )

        pdbstr = to_pdb(protein)

        return pdbstr

    def sample(self) -> List[str]:
        """
        サンプルを生成する関数
        Returns:
            List[str]: 生成されたサンプルのリスト
        """

        pdbstr = self._get_pdbstr(self.sequences[0])

        with open(self.pdb_path, "w") as f:
            f.write(pdbstr)

        samples = []
        records = []

        for _ in range(self.num_samples):
            wt_sequence = random.choice(self.sequences)
            mutant_sequence = self._mutate(wt_sequence)
            samples.append(mutant_sequence)
            records.append(SeqRecord(Seq(mutant_sequence), id="sample", description=""))

        SeqIO.write(records, self.sample_path, "fasta")

        return samples

    def load(self) -> List[str]:
        """
        Returns:
            List[str]: サンプルのリスト
        """
        samples = []
        records = SeqIO.parse(self.sample_path, "fasta")
        for record in records:
            samples.append(str(record.seq))
        return samples
