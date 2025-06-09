from __future__ import annotations

from transformers import AutoModel
import torch
import torch.nn as nn


class Regressor(nn.Module):
    """
    回帰モデルを定義するクラス
    """

    def __init__(self, backbone: AutoModel, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.regression_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_ids (torch.Tensor, shape (B, L)): トークン列
            attention_mask (torch.Tensor, shpae (B, L)): マスク
            labels (torch.Tensor, optional, shape (B, 1)): 正解ラベル

        Returns:
            dict[str, torch.Tensor]: {"loss", "logits"} の辞書
        """
        output = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        mask = (
            attention_mask.unsqueeze(-1).expand(output.last_hidden_state.size()).float()
        )
        pooled = output.last_hidden_state.masked_fill(mask == 0, -1e9).max(1).values
        logits = self.regression_head(pooled)
        loss = nn.MSELoss()(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}
