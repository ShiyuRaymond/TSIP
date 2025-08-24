import torch
from torchmetrics import Metric
from typing import List, Optional, Any, Callable
from pycocoevalcap.cider.cider import Cider
import statistics
from yoloworld_lightning.utils.nlp_metrics.cocval_evalution import eval_nlp_scores
# from cocval_evalution import eval_nlp_scores
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import evaluate
from pycocoevalcap.cider.cider import Cider



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

        self.bleu1 = torch.tensor(0.0)
        self.bleu2 = torch.tensor(0.0)
        self.bleu3 = torch.tensor(0.0)
        self.bleu4 = torch.tensor(0.0)
        self.cider = torch.tensor(0.0)
        self.meteor = torch.tensor(0.0)
        self.rougel = torch.tensor(0.0)
        self.harmonice = torch.tensor(0.0)

    def update(self, preds: List[str], targets: List[str]):
        """
        传入一批原始文本字符串
        - preds: List[str]，预测的caption
        - targets: List[str]，真实caption
        """
        self.preds.extend(preds)
        self.targets.extend(targets)

    def compute(self):
        # references = [[ref] for ref in self.targets]  # list of list of refs
        references = self.targets
        hypotheses = self.preds
        # print(f'references :{references}' )
        # print(f'hypotheses {hypotheses}')
        
        # try:
        metrics_dict = eval_nlp_scores(hypotheses, references)
        # except Exception as e:
        #     print("[❌ NLP Metric Compute Error]")
        #     print("Exception:", e)
        #     print("Hypotheses:")
        #     for k, v in hypotheses.items():
        #         print(f"  {k}: {v}")
        #     print("References:")
        #     for k, v in references.items():
        #         print(f"  {k}: {v}")
        #     raise e  # 继续抛出异常以不中断主流程
        self.bleu1 = torch.tensor(metrics_dict['Bleu_1'][0])
        self.bleu2 = torch.tensor(metrics_dict['Bleu_2'][0])
        self.bleu3 = torch.tensor(metrics_dict['Bleu_3'][0])
        self.bleu4 = torch.tensor(metrics_dict['Bleu_4'][0])
        self.cider = torch.tensor(self.compute_cider(self.preds, self.targets))
        self.meteor = torch.tensor(self.compute_meteor(self.preds, self.targets))
        self.rougel = torch.tensor(metrics_dict['ROUGE_L'][0])
        self.harmonice = torch.tensor(statistics.harmonic_mean([
            metrics_dict['Bleu_4'][0], float(self.meteor)
        ]))

        return {
            "bleu1": self.bleu1,
            "bleu2": self.bleu2,
            "bleu3": self.bleu3,
            "bleu4": self.bleu4,
            "meteor": self.meteor,
            "rougel": self.rougel,
            "cider": self.cider,
            "harmonic_mean": self.harmonice
        }

    def reset(self):
        self.preds.clear()
        self.targets.clear()
        self.bleu1.zero_()
        self.bleu2.zero_()
        self.bleu3.zero_()
        self.bleu4.zero_()
        self.cider.zero_()
        self.meteor.zero_()
        self.rougel.zero_()
        self.harmonice.zero_()
        
    def compute_meteor(self, preds, refs):
        """
        计算 METEOR 分数（已修复批量输入问题）
        - preds: List[str]，预测句子
        - refs: List[List[str]]，每个预测句子对应的参考句子列表
        """
        assert len(preds) == len(refs), "预测和参考数量不一致"

        scores = []
        for pred, ref_list in zip(preds, refs):
            try:
                tokenized_pred = word_tokenize(pred)  # ⬅️ 这里是字符串
                tokenized_refs = [word_tokenize(ref) for ref in ref_list]  # ⬅️ 每个参考句也分词
                score = meteor_score(tokenized_refs, tokenized_pred)
            except Exception as e:
                print(f"[METEOR ERROR] pred: {pred}, refs: {ref_list}, error: {e}")
                score = 0.0
            scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0
    def compute_cider(self, preds: List[str], refs: List[List[str]]) -> float:
        """
        计算 CIDEr 分数
        - preds: List[str]，模型预测句子
        - refs:  List[List[str]]，每个预测对应的参考句子列表
        返回：
        float，全局的 CIDEr 分数
        """
        assert len(preds) == len(refs), "预测和参考数量不一致"

        # 构造 COCO 格式输入
        gts = {}
        res = {}
        for i, (pred, ref_list) in enumerate(zip(preds, refs)):
            # 确保都是字符串
            p = pred if isinstance(pred, str) else ""
            r_list = [r if isinstance(r, str) else "" for r in ref_list]
            # gts expects a list of reference strings, res expects a one-element list of pred
            gts[i]  = r_list
            res[i]  = [p]

        scorer = Cider()
        corpus_score, _ = scorer.compute_score(gts, res)
        return float(corpus_score)
    
    def compute_bleu(self, preds, gts, n_gram=4):
        """
        preds: List[str]，预测句子列表
        gts: List[str]，真实句子列表（长度同preds）
        return: List[float]，每句BLEU分数
        """
        smoothie = SmoothingFunction().method1
        scores = []
        for pred, gt in zip(preds, gts):
            # 分词
            pred_tokens = nltk.word_tokenize(pred.lower())
            gt_tokens = nltk.word_tokenize(gt.lower())
            # BLEU-n
            weights = tuple([1.0/n_gram]*n_gram)
            score = sentence_bleu(
                [gt_tokens], pred_tokens, weights=weights, smoothing_function=smoothie
            )
            scores.append(score)
        return scores
    
    def nlp_metric_cider(self, preds, gts):
        scorer = Cider()

        """
        preds: List[str]，预测句子列表
        gts: List[str]，真实句子列表（长度同preds）
        return: List[float]，每句CIDEr分数
        """
        # 按照coco-caption要求，输入dict，id需唯一
        hypo = {i: [sent] for i, sent in enumerate(preds)}
        refs = {i: gts[i] for i in range(len(preds))}

        scorer = Cider()
        (score, scores) = scorer.compute_score(refs, hypo)
        # score: 平均分
        # scores: 每个样本的分数

        return score, scores
        
    
    
    

