import torch
import torch.nn as nn
import torch.nn.functional as F


import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import AutoModel,AutoTokenizer, AutoModelForCausalLM, AutoConfig  
from transformers.models.opt import OPTForCausalLM
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from transformers import BertForMaskedLM

from transformers import get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup


from MSR_VTT_MODEL.video_caption import YOLOWorldCaptioning
from utils.nlp_metrics.NLP_metrics import  nlp_metric_bert

from utils.track_cuda import track_cuda_memory
from utils.loss import  SoftPromptCenterLoss, soft_prompt_cosine_diversity_loss,embedding_align_loss, grammar_error_loss,no_repeat_ngram_loss, ending_ngram_match_loss,compute_mlm_loss, compute_align_loss, compute_svo_loss,scheduled_sampling, simclr_infonce_loss
import pickle 
from omegaconf import OmegaConf
import video_clip.video_clip as video_clip  # Import the video_clip package


from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

import torch
import os
import random
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge

import re 

import language_tool_python

# 1. 初始化英文纠错器
tool = language_tool_python.LanguageTool('en-US')

import re

def remove_consecutive_duplicates(text):
    # 去除连续重复的单词，例如 "man man is" -> "man is"
    return re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

def ensure_complete_sentence(text):
    # 若句子缺少谓语动词，可以尝试补全（简单启发式）
    verbs = ["is", "are", "was", "were", "has", "have", "does", "do", "can", "will"]
    if not any(verb in text.lower().split() for verb in verbs):
        # 简单添加一个默认谓语
        return text + " is happening"
    return text

def polish_caption_for_rouge(text, lowercase=True):
    # 去除首尾空格
    text = text.strip()

    # 1. 去重
    text = remove_consecutive_duplicates(text)

    # 2. 补全谓语（可选）
    # text = ensure_complete_sentence(text)

    # 3. 标点统一（句末加句号）
    # if not text.endswith(('.', '!', '?')):
    #     text += '.'

    # # 4. 大小写处理
    # if lowercase:
    #     text = text.lower()
    # else:
    #     text = text[0].upper() + text[1:]

    return text



def set_requires_grad(module, flag: bool):
    """递归修改一个 module 里的所有参数的 requires_grad"""
    for p in module.parameters():
        p.requires_grad = flag


def unfreeze_llm_last_n_layers(llm, n_last: int = 2):
    """
    仅解冻 decoder 的最后 n_last 层。
    适配常见 HuggingFace decoder-only 或 encoder-decoder 模型。
    """
    total_layers = llm.config.num_hidden_layers
    target_ids = {total_layers - i - 1 for i in range(n_last)}  # 末 N 层 idx
    for name, p in llm.named_parameters():
        if any(f"layers.{idx}." in name for idx in target_ids):
            p.requires_grad = True

def strip_prefix(pred: str, prefix="This video shows:"):
    if pred.lower().startswith(prefix):
        return pred[len(prefix):].strip()
    return pred



class TSIP(pl.LightningModule):
    def __init__(self, 
                 cfg,
                 lr: float = 1e-4, 
                 dropout: float = 0.1, 
                 grad_clip: float = 1.0,
                 ):
        
        """
        stage: 1 表示软提示匹配训练阶段，2 表示描述生成阶段。
        lr: 学习率。
        dropout: 模型中使用的 dropout 概率（用于日志记录）。
        grad_clip: 梯度裁剪值（用于日志记录）。
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = YOLOWorldCaptioning(cfg.base_model)  # 实例化底层模型
        self.stage = cfg.stage
        self.max_length = cfg.model.max_length
        self.eval_config = cfg.model.eval_config

        # 第二阶段，默认解冻需要微调的模块, 使用llm进行训练
        self.accuracy = nlp_metric_bert()
        
        
        # 加载LLM和Tokenizer
        llm_config = AutoConfig.from_pretrained("facebook/bart-base")

        self.llm = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        self.video_encoder  =  video_clip.load_finetune_model(self.eval_config)
        
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").eval()
        self.gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        # self.sentence_bert = SentenceTransformer('paraphrase-mpnet-base-v2')


        # model_name = "prithivida/grammar_error_correcter_v1"
        # model_name = "vennify/t5-base-grammar-correction"

        # self.grammer_tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.grammer_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).eval()

        
        # self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        
        self.unfreeze_llm_last_n = 2
        self.unfreeze_mae_n = 2
        self.unfreeze_qformer = 2
        self.unfreze_llm_encoder_n = 2
        self._configure_freeze_policy(self.unfreeze_llm_last_n, self.unfreeze_mae_n, self.unfreeze_qformer,self.unfreze_llm_encoder_n)
        self.video_ids = []


        self.alignment = SoftPromptCenterLoss()   # or "mse", "combine"

        self.cider_scorer = Cider()
        self.rouge_scorer = Rouge()




    def compute_cider_reward(self, predictions, references):
        """
        predictions: 模型生成的caption列表, 长度为batch_size
        references: 动态选取的ground-truth caption列表, 长度为batch_size
        返回: torch.Tensor (batch_size,) CIDEr得分
        """
        gts = {i: [ref] for i, ref in enumerate(references)}   # reference格式为字典形式
        res = {i: [pred] for i, pred in enumerate(predictions)}  # prediction也为字典形式
        cider_score, _ = self.cider_scorer.compute_score(gts, res)
        cider_scores_tensor = torch.tensor(cider_score, device=self.device, dtype=torch.float32)

         # 2. ROUGE 分数（pycocoevalcap的rouge是F1，如果你只想要Recall，可以再处理）
        rouge_score, scores = self.rouge_scorer.compute_score(gts, res)  # rouge_score 是平均，scores 是列表
        rouge_tensor = torch.tensor(scores, device=self.device, dtype=torch.float32)
       
        return 0.4* cider_scores_tensor + 0.6*rouge_tensor 
    
    def _configure_freeze_policy(self, unfreeze_llm_last_n=2, unfreeze_mae_n=2, unfreeze_qformer = 2, unfreeze_llm_enc_last_n=1):
        """
        统一冻结策略：
        • 全部 freeze
        • 指定模块解冻
        • 可选：LLM 后 N 层解冻
        """
        # 1) 冻结全模型
        set_requires_grad(self, False)

        # 2) 必须解冻的视觉→LLM桥接
        

        # 3) 若需要训练 early fusion / projector，可在此解冻
        #    示范：保持冻结 -> 不做操作
        
        set_requires_grad(self.model.vis_proj, True)
        set_requires_grad(self.model.fusion_module, True)
        set_requires_grad(self.model.detector.neck.text_enhancer, True)
        # set_requires_grad(self.model.detector.neck.top_down_layers, True)
        # set_requires_grad(self.model.detector.neck.downsample_layers, True)
        # set_requires_grad(self.model.detector.neck.bottom_up_layers, True)
        set_requires_grad(self.model.projector, True)
        
        # set_requires_grad(self.sentence_bert, False)
        
        set_requires_grad(self.gpt2, False)
        
        set_requires_grad(self.video_encoder.vision_proj, True)
        
        set_requires_grad(self.llm.model.decoder.embed_tokens, True)
        set_requires_grad(self.llm.model.decoder.layernorm_embedding, True)
        set_requires_grad(self.llm.model.decoder.embed_positions, True)

        # self.video_encoder.query_tokens.requires_grad = True
        # # 3) 解冻 LLM 的 decoder 最后2层
        total_dec_layers = self.llm.config.decoder_layers
        dec_target_ids = {total_dec_layers - i - 1 for i in range( unfreeze_llm_last_n )}
        for name, param in self.llm.model.decoder.named_parameters():
            if any(f"layers.{i}." in name for i in dec_target_ids):
                param.requires_grad = True
                
        total_enc_layers = self.llm.config.encoder_layers
        enc_target_ids = {total_enc_layers - i - 1 for i in range(unfreeze_llm_enc_last_n)}
        for name, param in self.llm.model.encoder.named_parameters():
            if any(f"layers.{i}." in name for i in enc_target_ids):
                param.requires_grad = True

        # 解冻最后2层
        total_videoencoder_layers = len(self.video_encoder.video_Qformer.bert.encoder.layer)
        dec_target_ids = {total_videoencoder_layers - i - 1 for i in range(unfreeze_mae_n)}
        for name, param in self.video_encoder.video_Qformer.bert.encoder.layer.named_parameters():
            if any(f"layers.{i}." in name for i in dec_target_ids):
                param.requires_grad = True
                
        total_Qformerencoder_layers = len(self.video_encoder.Qformer.bert.encoder.layer)
        dec_target_ids = {total_Qformerencoder_layers - i - 1 for i in range(unfreeze_qformer)}
        for name, param in self.video_encoder.Qformer.bert.encoder.layer.named_parameters():
            if any(f"layers.{i}." in name for i in dec_target_ids):
                param.requires_grad = True
        
        
        # num_layers = len( self.video_encoder.video_Qformer.bert.encoder.layer)
        # n_unfreeze = unfreeze_mae_n
        # for i in range(num_layers - n_unfreeze, num_layers):
        #     for param in  self.video_encoder.video_Qformer.bert.encoder.layer[i].parameters():
        #         param.requires_grad = True

        # 4) 解冻 LLM 的 encoder 最后2层
        # total_enc_layers = self.llm.config.encoder_layers
        # enc_target_ids = {total_enc_layers - i - 1 for i in range(2)}
        # for name, param in self.llm.model.encoder.named_parameters():
        #     if any(f"layers.{i}." in name for i in enc_target_ids):
        #         param.requires_grad = True

        # # 5) 解冻 lm_head（用于 MLM）
        # set_requires_grad(self.llm.lm_head, True)
        # =====================================================

    def setup(self, stage=None):
        if not hasattr(self, "tokenizer"):
            self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    
    def compute_gpt2_ppl(self, sent_list):
        # sent_list: list of strings
        scores = []
        for sent in sent_list:
            enc = self.gpt2_tokenizer(sent, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.gpt2(**enc, labels=enc["input_ids"])
                loss = outputs.loss.item()
                scores.append(loss)
        return scores  # loss越低越流畅
    
    def repetition_penalty_reward(self, caption):
        # 惩罚重复片段，例如3-gram重复
        tokens = caption.split()
        seen = set()
        for i in range(len(tokens)-2):
            ngram = tuple(tokens[i:i+3])
            if ngram in seen:
                return 0.0
            seen.add(ngram)
        return 1.0
    
    

    @torch.no_grad()
    def generate_caption_from_cross_attn(
        self,
        soft_prompt: torch.Tensor,          # [B, L, D] 来自 QFormer 输出
        prompt_text: str = "This video shows:",    # decoder 的起始提示
        max_new_tokens: int = 64,
        min_length =6,
        num_beams: int = 8,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        # top_p: float = 0.79,
        no_repeat_ngram_size=2,
        repetition_penalty=1.5,
        length_penalty=0.7,
    ):
        """
        使用 encoder-decoder 架构（BART）+ soft prompt（QFormer输出）生成 caption。
        """
        B = soft_prompt.size(0)
        device = soft_prompt.device

        # 1. 准备 decoder 起始输入（如 "a video of: : "）
        decoder_input_ids = self.tokenizer(
            [prompt_text] * B,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True
        ).input_ids.to(device)  # [B, T]

        # 2. 构造 encoder attention mask（soft_prompt 是 encoder_outputs）
        # encoder_attention_mask = torch.ones(soft_prompt.shape[:-1], dtype=torch.long).to(device)  # [B, L]

        # 3. 构造 encoder_outputs（模拟 Encoder 的输出结构）
        encoder_outputs = BaseModelOutput(last_hidden_state=soft_prompt)

        # 获取prompt的token数，用于修正min_length
        prompt_token_len = decoder_input_ids.shape[1]
        # min_length 应设置为 prompt+8，保证新生成部分≥8
        effective_min_length = prompt_token_len + min_length


        # 4. 调用 generate（Cross-Attn 注入 soft prompt）
        generated_ids = self.llm.generate(
            input_ids=decoder_input_ids,                         # decoder 的初始 token
            encoder_outputs=encoder_outputs,                     # 👈 视觉 soft prompt 注入点
            # encoder_attention_mask=encoder_attention_mask,       # 对应位置的 mask
            max_new_tokens=max_new_tokens,
            min_length = min_length,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            no_repeat_ngram_size = no_repeat_ngram_size,
            top_k=top_k,
            # top_p=top_p,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_beams,
            # num_beam_groups=2,      # 分组提升多样性
            # diversity_penalty=0.3
        )
        # 5. 解码生成结果
        captions = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        candidates = [captions[i:i+num_beams] for i in range(0, len(captions), num_beams)]  

        try:  
            final_caps, _ = zip(*[self.rerank_candidates([c for c in cand_list if c.strip()]) for cand_list in candidates])  # 按batch

        except:
            print(candidates)
        
        return final_caps
    
    @torch.no_grad()
    def gpt2_score(self, caption):
        # GPT2得分越高，说明越流畅/natural（通常用负对数似然/perplexity）
        tokens = self.gpt2_tokenizer.encode(caption, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.gpt2(tokens, labels=tokens)
            loss = outputs.loss.item()
        # 注意，loss越小，说明模型越认可
        return -loss
    
    @torch.no_grad()
    def gpt2_local_ngram_reward(self, caption, n=4):
        """
        caption: str
        n: int, n-gram窗口长度（如4）
        返回: 所有n-gram的gpt2得分平均值
        """
        tokens = caption.strip().split()
        if len(tokens) < n:
            # 短句直接用整体score
            return self.gpt2_score(caption)
        scores = []
        for i in range(len(tokens) - n + 1):
            ngram = " ".join(tokens[i:i+n])
            score = self.gpt2_score(ngram)
            scores.append(score)
        return sum(scores) / len(scores)

    # @torch.no_grad()
    # def grammar_correct(self, sentence, max_new_tokens=64):
    #     # 直接输入错句，不需要加指令prompt
    #     input_ids = self.grammer_tokenizer.encode(sentence, return_tensors='pt').to(self.device)
    #     with torch.no_grad():
    #         outputs = self.grammer_model.generate(
    #             input_ids=input_ids,
    #             max_new_tokens=64,         # 通常足够，长句可适当加大
    #             num_beams=4,               # 4-5个beam比1更鲁棒，易出最优结果，几乎不影响速度
    #             do_sample=False,           # 不采样，结果更稳定
    #             early_stopping=True,       # 优先停止
    #             use_cache=False,           # 必须加，避免cache相关bug
    #             no_repeat_ngram_size=3,    # 防止重复，提升句子质量
    #             repetition_penalty=1.1,    # 稍微惩罚重复
    #         )
    #     result = self.grammer_tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     return result.strip()
    
    1

        # soft_emb = soft_promts.mean(dim=0)
        # soft_emb = F.normalize(soft_emb, dim=-1).unsqueeze(0)  # [1, D]

        # # 2. 句子embedding，shape [N, D]
        # with torch.no_grad():
        #     caption_embs = self.sentence_bert.encode(
        #         candidates, convert_to_tensor=True, device=self.device)
        # caption_embs = F.normalize(caption_embs, dim=-1)  # [N, D]

        # # 3. 计算余弦相似度
        # sims = torch.matmul(caption_embs, soft_emb.t()).squeeze(-1)  # [N]

        # # 4. 排序并返回
        # sorted_idx = torch.argsort(sims, descending=True)
        # ranked_list = [(candidates[i], sims[i].item()) for i in sorted_idx]
        # best_caption = ranked_list[0][0]

        # return best_caption, ranked_list



    def on_fit_start(self):
        if self.stage == 2 :
            self.accuracy.to(self.device)

    def forward(self, frames):
        """将调用底层模型的前向传播。"""
        # 1. 计算采样索引
        
        # 2. 按照dim=1采样
        # indices = torch.linspace(0, 48 - 1, steps=32).long()
        video_tokens = self.video_encoder.encode_videoQformer_visual(frames)[-1].last_hidden_state
        # video_embeddings = F.normalize(self.video_encoder.vision_proj(video_tokens), dim=-1) #[B, T, C]
        video_embeddings = self.video_encoder.vision_proj(video_tokens) #[B, T, C]
        fused_embeddings, soft_prompt = self.model(frames, video_embeddings)     # [B, embed_dim]     
        return fused_embeddings, soft_prompt

    def eos_position(self, ids, eos_token_id):
        # ids: [B, T]
        # 返回每个序列中第一个eos token的位置，没有则为T
        pos = []
        for row in ids.tolist():
            if eos_token_id in row:
                pos.append(row.index(eos_token_id))
            else:
                pos.append(len(row))
        return torch.tensor(pos, device=ids.device)

    def bart_encode(self, text_list):
        # 返回每个句子的 mean pooling encoder embedding
        inputs = self.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True, max_length=64)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # with torch.no_grad():
        outputs = self.llm.model.encoder(**inputs)
        last_hidden = outputs.last_hidden_state  # [batch, seq, hidden]
        # 对每个句子做 mean pooling
        mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = summed / counts  # [batch, hidden]
        return mean_pooled

    def training_step(self, batch, batch_idx):
        frames, captions, video_ids = batch
        fused_emd, soft_prompt = self(frames)  # 只用图像，文本等下处理

        noise_std = 0.01
        # 显式 detach，避免多次使用造成 autograd graph 堆积
        fused_feature_1 = fused_emd + torch.randn_like(fused_emd) * noise_std
        fused_feature_2 = fused_emd + torch.randn_like(fused_emd) * noise_std
        soft_prompt_1 = self.model.vis_proj(fused_feature_1)
        soft_prompt_2 = self.model.vis_proj(fused_feature_2)
        infor_loss = simclr_infonce_loss(soft_prompt_1, soft_prompt_2)
        infor_loss_val = infor_loss.item()


        # Step 3: Dynamic Caption Selection
        selected_captions = []

        for idx in range(len(frames)):
            caption = captions[idx]  # list of multiple captions
            caption_lengths = torch.tensor([len(cap.split()) for cap in caption], device=self.device)
            target_length = caption_lengths.float().mean().item()  # dynamically set to average length

            with torch.no_grad():
                caption_embeddings = self.bart_encode(caption)  # [num_captions, hidden_dim]

            video_embedding = soft_prompt[idx].mean(dim=0, keepdim=True)  # [1, hidden_dim]

            sim_scores = F.cosine_similarity(video_embedding, caption_embeddings, dim=-1)  # [num_captions]

            length_scores = torch.exp(-((caption_lengths - target_length)**2) / (2*(2**2)))

            total_scores = 0.7 * sim_scores + 0.5 * length_scores
            top_k_indices = total_scores.topk(min(5, len(captions))).indices
            selected_index = top_k_indices[torch.randint(len(top_k_indices), (1,)).item()].item()
            # selected_index = total_scores.argmax().item()
            selected_captions.append(caption[selected_index])


        """训练过程中的每一步。根据阶段计算不同的损失，并记录日志。"""
        # 第二阶段：计算生成文本与真实文本的交叉熵损失
        torch.cuda.reset_peak_memory_stats()  # 💡 重置显存峰值统计

        # 1. 构造 hard prompt：自然语言提示
    
        # 2. 构造 ground-truth captions
        caption_tokenized = self.tokenizer(selected_captions, padding="max_length", truncation=True, max_length=32, return_tensors="pt").to(self.device)
        
        caption_ids = caption_tokenized.input_ids.to(self.device)
        
        labels = caption_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100         # pad -> -100
        encoder_outputs = BaseModelOutput(last_hidden_state=soft_prompt)
        mlm_inputs = self.tokenizer(selected_captions, return_tensors='pt', padding=True, truncation=True, max_length=16).to(self.device)
        mlm_labels = mlm_inputs.input_ids.clone()

        probability_matrix = torch.full(mlm_labels.shape, 0.15).to(labels.device)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        mlm_labels[~masked_indices] = -100
        mlm_inputs.input_ids[masked_indices] = self.tokenizer.mask_token_id

        mlm_outputs = self.llm(**mlm_inputs, labels=mlm_labels)
        mlm_loss = mlm_outputs.loss 
        mlm_loss_val = mlm_loss.item()
        
        
        # 使用hidden state策略
        outputs = self.llm(
            input_ids=caption_ids,
            labels=labels,
            encoder_outputs=encoder_outputs
        )

        # 使用scheduled sampling后的loss
        caption_loss = outputs.loss  
        caption_loss_val = caption_loss.item()

        with torch.no_grad():
            caption_embeds = self.llm.model.encoder(input_ids=caption_ids).last_hidden_state.to(self.device)

        alignment_loss = self.alignment(F.normalize(soft_prompt.mean(dim=1), dim=-1), F.normalize(caption_embeds.mean(dim=1), dim=-1) )
        alignment_loss_val = alignment_loss.item()
        
        # Step 6: Reinforcement Learning Loss
        sampled_caps = self.generate_caption_from_cross_attn(soft_prompt, num_beams=4, repetition_penalty=2.0, length_penalty=0.9)
        cider_reward = self.compute_cider_reward(sampled_caps, selected_captions)
        rl_loss = -torch.mean(cider_reward)
        rl_loss_val = rl_loss.item()
        
        diversity_loss = soft_prompt_cosine_diversity_loss(soft_prompt)
        diversity_loss_val = diversity_loss.item()
        loss = caption_loss +   infor_loss  + 0.2 * rl_loss + 0.2 * diversity_loss
        total_loss_val = loss.item()
        
        # 5. 计算 loss

        # 记录训练损失
        self.log("train/infor_loss", infor_loss_val, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/rl_loss", rl_loss_val, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/diversity_loss", diversity_loss_val, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/mlm_loss", mlm_loss_val, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/alignment_loss", alignment_loss_val, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/caption_loss", caption_loss_val, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/train_loss", total_loss_val, on_step=True, on_epoch=True, prog_bar=True)
        
        

        # 5. 显存统计日志
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
        self.log("train/peak_mem_MB", peak_mem, prog_bar=True, on_epoch=True)
        # Step 5: 显式释放防泄漏
        del outputs, soft_prompt, caption_ids, labels, caption_loss, infor_loss
        torch.cuda.empty_cache()
        
        
        return loss
   
   
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """验证过程：生成描述并收集预测和真实值，用于计算指标。"""
        frames_tensor, caption ,video_id  = batch
        
        # 1. 生成 caption（预测）
        fused_emd,soft_prompts = self(frames_tensor)
        
        # tem_res = log_per_prompt_statistics(soft_prompts)
        
        # pask key val 策略 1
        generated_captions = self.generate_caption_from_cross_attn(
            soft_prompt=soft_prompts
        )
        
        generated_captions = [strip_prefix(p) for p in generated_captions]
        generated_captions = [polish_caption_for_rouge(cap) for cap in generated_captions]
        # 2. 更新 NLP-metrics（BLEU, METEOR, CIDEr 等）
        self.accuracy.update(generated_captions, caption)
        
        if batch_idx == 0 and self.trainer.is_global_zero:
            # for i, (pred, ref) in enumerate(zip(generated_captions, caption)):
            #     if i >= 2:  # 最多记录 2 个样本，防止日志过多
            #         break
            #     self.logger.experiment.add_text(
            #         tag=f"val_caption_epoch{self.current_epoch}_sample{i}",
            #         text_string=f"**GT:** {ref[:10]}\n**Pred:** {pred}",
            #         global_step=self.global_step
            #     )

            # 随机选2个样本
            idxs = random.sample(range(len(generated_captions)), min(2, len(generated_captions)))
            for j, i in enumerate(idxs):
                pred = generated_captions[i]
                ref = caption[i]
                vid = video_id[i]
                self.logger.experiment.add_text(
                    tag=f"val_caption_epoch{self.current_epoch}_sample{j}_vid{vid}",
                    text_string=f"**video_id:** {vid}\n**GT:** {ref[:10]}\n**Pred:** {pred}",
                    global_step=self.global_step
                )
        
        del fused_emd, soft_prompts, generated_captions
        torch.cuda.empty_cache()
    
    @torch.no_grad()
    def on_validation_epoch_end(self):
        score_dict = self.accuracy.compute()  # self.val 是验证集的 reference caption 数据

        # 分别记录各项指标
        self.log('val/harmonic_mean_bleu_meteor', self.accuracy.harmonice, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/bleu1', self.accuracy.bleu1, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/bleu2', self.accuracy.bleu2, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/bleu3', self.accuracy.bleu3, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/bleu4', self.accuracy.bleu4, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/meteor', self.accuracy.meteor, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/cider', self.accuracy.cider, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/rougel', self.accuracy.rougel, on_epoch=True, prog_bar=True, logger=True)
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
        self.log("val/peak_mem_MB", peak_mem, prog_bar=True, on_epoch=True)

        # 重置指标缓存
        self.accuracy.reset()

    
    def test_step(self, batch, batch_idx):
        cider_metric = self.accuracy.nlp_metric_cider

        """测试过程：生成描述并收集预测和真实值，用于计算指标。"""
        frames_tensor, caption, video_ids = batch
        # 1. 生成 caption（预测）
        fused_emd,soft_prompts = self(frames_tensor)        
        generated_captions = self.generate_caption_from_cross_attn(
            soft_prompt=soft_prompts
        )
        
        generated_captions = [strip_prefix(p) for p in generated_captions]
        generated_captions = [polish_caption_for_rouge(cap) for cap in generated_captions]
        # self.update_json('/workspace/YOLO-World/scripts/yoloworld_lightning/data/msrvtt/soft_prompt',video_ids, soft_prompts)
        output_dir = '/workspace/YOLO-World/scripts/yoloworld_lightning/data/msrvtt/txt'
        os.makedirs(output_dir, exist_ok=True)

        for vid, cap in zip(video_ids, generated_captions):
            txt_path = os.path.join(output_dir, f"{vid}.txt")
            # 如果想每个文件只保存一条caption：
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(cap.strip() + "\n")

        # _, baseline_cider_scores = cider_metric(generated_captions, caption)
        # for i, c in enumerate(baseline_cider_scores):
        #     if c < 0.55:
        #         self.video_ids.append(i)
                
        # 2. 更新 NLP-metrics（BLEU, METEOR, CIDEr 等）
        self.accuracy.update(generated_captions, caption)
        
        if batch_idx == 0 and self.trainer.is_global_zero:
            for i, (pred, ref) in enumerate(zip(generated_captions, caption)):
                if i >= 2:  # 最多记录 2 个样本，防止日志过多
                    break
                self.logger.experiment.add_text(
                    tag=f"val_caption_epoch{self.current_epoch}_sample{i}",
                    text_string=f"**GT:** {ref}\n**Pred:** {pred}",
                    global_step=self.global_step
                )
        
        del fused_emd, soft_prompts, generated_captions
        torch.cuda.empty_cache()
    
    
    def on_test_epoch_end(self):
        # print(self.video_ids)
        # 1. 计算所有 NLP 指标
        score_dict = self.accuracy.compute()

        # 2. 记录每个指标
        self.log('test/harmonic_mean_bleu_meteor', torch.tensor(self.accuracy.harmonice,device=self.device), on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        self.log('test/bleu1', torch.tensor(self.accuracy.bleu1,device=self.device), on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        self.log('test/bleu2', torch.tensor(self.accuracy.bleu2,device=self.device), on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        self.log('test/bleu3', torch.tensor(self.accuracy.bleu3,device=self.device), on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        self.log('test/bleu4', torch.tensor(self.accuracy.bleu4,device=self.device), on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        self.log('test/meteor', torch.tensor(self.accuracy.meteor,device=self.device), on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        self.log('test/cider', torch.tensor(self.accuracy.cider,device=self.device), on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        self.log('test/rougel', torch.tensor(self.accuracy.rougel,device=self.device), on_epoch=True, prog_bar=True, logger=True,sync_dist=True)

        # 可选：记录汇总指标
        # self.log("test/harmonic_mean_bleu_meteor", score_dict["harmonic_mean"], prog_bar=True, logger=True)

        # 3. 清空缓存
        self.accuracy.reset()


    def update_json(self, output_dir, video_ids, soft_prompt):
        os.makedirs(output_dir, exist_ok=True)

        for vid, caption in zip(video_ids, soft_prompt):
            # 保证 vid 是字符串
            if not isinstance(vid, str):
                vid = str(vid)
            # 构造路径
            out_path = os.path.join(output_dir, f"{vid}.pt")
            # 写入文件
            torch.save(caption, out_path)

    
            
    def configure_optimizers(self):
        """根据阶段返回相应的优化器（和学习率调度器）。"""
            
        # 1. 收集各组参数
        vis_proj_params     = list(self.model.vis_proj.parameters())
        fusion_params = list( self.model.fusion_module.parameters())
        text_enhancer_params = list(self.model.detector.neck.text_enhancer.parameters())
        projoctor_params = list(self.model.projector.parameters())
        word_embedding_params = list(self.llm.model.decoder.embed_tokens.parameters())
        position_embedding_params = list(self.llm.model.decoder.embed_positions.parameters())

        # qformer_tokens_params = [self.video_encoder.query_tokens]
        decoder_layernorm = list(self.llm.model.decoder.layernorm_embedding.parameters())
        # 2. 解冻 LLM 的 decoder 最后2层
        dec_params = []
        total_dec_layers = self.llm.config.decoder_layers
        dec_target_ids = {total_dec_layers - i - 1 for i in range( self.unfreeze_llm_last_n )}
        for name, param in self.llm.model.decoder.named_parameters():
            if any(f"layers.{i}." in name for i in dec_target_ids):
                dec_params.append(param)
        enc_params = []
        total_enc_layers = self.llm.config.encoder_layers
        enc_target_ids = {total_enc_layers - i - 1 for i in range(self.unfreze_llm_encoder_n)}
        for name, param in self.llm.model.encoder.named_parameters():
            if any(f"layers.{i}." in name for i in enc_target_ids):
                enc_params.append(param)
                
        video_encoder_params = []
        num_layers = len(self.video_encoder.video_Qformer.bert.encoder.layer)
        dec_target_ids = {num_layers - i - 1 for i in range(self.unfreeze_mae_n)}
        for name, param in self.video_encoder.video_Qformer.bert.encoder.layer.named_parameters():
            if any(f"layers.{i}." in name for i in dec_target_ids):
                video_encoder_params.append(param)
                
        video_encoder_qformer_params = []
        num_layers = len(self.video_encoder.Qformer.bert.encoder.layer)
        dec_target_ids = {num_layers - i - 1 for i in range(self.unfreeze_qformer)}
        for name, param in self.video_encoder.Qformer.bert.encoder.layer.named_parameters():
            if any(f"layers.{i}." in name for i in dec_target_ids):
                video_encoder_qformer_params.append(param)
               
        total_steps = int(self.trainer.estimated_stepping_batches)
        # image_pooler_params = list(self.model.detector.neck.downsample_layers.parameters())
        # cross_enhancer_params = list(self.model.detector.neck.bottom_up_layers.parameters())
        # top_down_layer_params = list(self.model.detector.neck.top_down_layers.parameters())
        
         
        # 3. 定义优化器，分别给不同组不同 lr
        optimizer = torch.optim.AdamW([
            {"params": vis_proj_params , "lr": 9e-6 , "weight_decay": 0.01},
            {"params": projoctor_params, "lr": 9e-6, "weight_decay": 0.01},
            {"params": fusion_params  , "lr": 9e-6, "weight_decay": 0.01},
            {"params": text_enhancer_params , "lr": 9e-6, "weight_decay": 0.01},
            {"params": dec_params , "lr": 9e-6, "weight_decay": 0.01},
            {"params":  word_embedding_params, "lr": 9e-6, "weight_decay": 0.01},
            {"params": decoder_layernorm, "lr": 9e-6, "weight_decay": 0.01},
            {"params": position_embedding_params, "lr": 9e-6, "weight_decay": 0.01},
            
            {"params": enc_params, "lr": 9e-6, "weight_decay": 0.01},
            {"params": video_encoder_params, "lr": 9e-6, "weight_decay": 0.01},  
            {"params": video_encoder_qformer_params, "lr": 9e-6, "weight_decay": 0.01},
        ])

        # 4. 学习率调度器（cosine+warmup 举例）
        scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.3* total_steps),  # 前10%步数用于warmup
                num_training_steps=int( total_steps),
            )       
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",   # 每step更新
                "frequency": 1,
            },
            "monitor": "val_cider",  # 这行其实可以不加，因为我们用cosine
        }
               
# def polish_english(sentence, tokenizer, model, max_new_tokens=40):
#     # 指令prompt，专为英文错句润色
#     prompt = (
#         "Please correct and rewrite the following sentence in fluent, grammatically correct English:\n"
#         f"{sentence}\n"
#         "Corrected:"
#     )
#     inputs = tokenizer(prompt, return_tensors="pt")
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=max_new_tokens,
#         do_sample=False,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.eos_token_id,
#         use_cache=False
#     )
#     result = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     # 可做后处理只提取Corrected:后面的内容
#     if "Corrected:" in result:
#         result = result.split("Corrected:")[-1].strip()
#     return result