import torch
from torchmetrics import Metric
from typing import List, Optional, Any, Callable
import evaluate


class nlp_metric_bert(Metric):
    def __init__(
        self,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn
        )

        self.preds = []
        self.targets = []

        # 加载 HuggingFace 的评估指标
        self.bleu = evaluate.load("bleu")
        self.meteor = evaluate.load("meteor")
        self.rouge = evaluate.load("rouge")
        # self.cider = evaluate.load("cider")

    def update(self, preds: List[str], targets: List[str]):
        """
        传入一批原始文本字符串
        - preds: List[str]，预测的caption
        - targets: List[str]，真实caption
        """
        self.preds.extend(preds)
        self.targets.extend(targets)

    def compute(self):
        refs = [[t] for t in self.targets]  # HuggingFace 需要 List[List[str]]

        bleu_score = self.bleu.compute(predictions=self.preds, references=refs)
        meteor_score = self.meteor.compute(predictions=self.preds, references=refs)
        rouge_score = self.rouge.compute(predictions=self.preds, references=refs)
        cider_score = self.cider.compute(predictions=self.preds, references=refs)

        # harmonic mean (BLEU4 和 METEOR)
        bleu4 = bleu_score["precisions"][3]
        meteor = meteor_score["meteor"]
        harmonic_mean = 2 * bleu4 * meteor / (bleu4 + meteor + 1e-8)

        return {
            "bleu1": torch.tensor(bleu_score["precisions"][0]),
            "bleu2": torch.tensor(bleu_score["precisions"][1]),
            "bleu3": torch.tensor(bleu_score["precisions"][2]),
            "bleu4": torch.tensor(bleu_score["precisions"][3]),
            "meteor": torch.tensor(meteor),
            "rougel": torch.tensor(rouge_score["rougeL"]),
            # "cider": torch.tensor(cider_score["score"]),
            # "harmonic_mean": torch.tensor(harmonic_mean)
        }

    def reset(self):
        self.preds.clear()
        self.targets.clear()
