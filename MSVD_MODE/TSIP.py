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


from MSVD_MODE.video_caption import YOLOWorldCaptioning
from utils.nlp_metrics.NLP_metrics import  nlp_metric_bert
from utils.track_cuda import track_cuda_memory
from utils.loss import compute_mlm_loss
import pickle 
from omegaconf import OmegaConf



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

def strip_prefix(pred: str, prefix="a video of "):
    if pred.lower().startswith(prefix):
        return pred[len(prefix):].strip()
    return pred



class TSIP(pl.LightningModule):
    def __init__(self, 
                 cfg,
                #  stage: int = 1, 
                 lr: float = 1e-4, 
                 dropout: float = 0.1, 
                 grad_clip: float = 1.0,
                #  temperature = 0.07,
                #  embed_dim = 512,
                #  llm_name = 'bert-base-uncased',
                #  max_length = 50,
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
                      
        # 第二阶段，默认解冻需要微调的模块, 使用llm进行训练
        self.accuracy = nlp_metric_bert()
        
        
        # 加载LLM和Tokenizer
        llm_config = AutoConfig.from_pretrained("facebook/bart-base")

        self.llm = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        # self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        
        self.unfreeze_llm_last_n = 2
        self._configure_freeze_policy(self.unfreeze_llm_last_n)
        
    def _configure_freeze_policy(self, unfreeze_llm_last_n=2):
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
        # set_requires_grad(self.model.scale_attention, True)
        # set_requires_grad(self.model.asymmetric_fusion_module, True)
        set_requires_grad(self.model.detector.neck.text_enhancer, True)
        set_requires_grad(self.model.projector, True)
        # # 3) 解冻 LLM 的 decoder 最后2层
        total_dec_layers = self.llm.config.decoder_layers
        dec_target_ids = {total_dec_layers - i - 1 for i in range( unfreeze_llm_last_n )}
        for name, param in self.llm.model.decoder.named_parameters():
            if any(f"layers.{i}." in name for i in dec_target_ids):
                param.requires_grad = True

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

    @torch.no_grad()
    def generate_caption_from_cross_attn(
        self,
        soft_prompt: torch.Tensor,          # [B, L, D] 来自 QFormer 输出
        prompt_text: str = "a video of",    # decoder 的起始提示
        max_new_tokens: int = 32,
        min_length = 6,
        num_beams: int = 5,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        no_repeat_ngram_size=3,
        
    ):
        """
        使用 encoder-decoder 架构（BART）+ soft prompt（QFormer输出）生成 caption。
        """
        B = soft_prompt.size(0)
        device = soft_prompt.device

        # 1. 准备 decoder 起始输入（如 "a video of"）
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
            # top_k=top_k,
            # top_p=top_p,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            length_penalty=1.2,
            repetition_penalty=1.1
            
        )

        # 5. 解码生成结果
        captions = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        return captions


    
    def on_fit_start(self):
        if self.stage == 2 :
            self.accuracy.to(self.device)

    def forward(self, frames, video_embedings):
        """将调用底层模型的前向传播。"""
        fused_embeddings, soft_prompt = self.model(frames, video_embedings)     # [B, embed_dim]     
        return fused_embeddings, soft_prompt


    def training_step(self, batch, batch_idx):
        frames, captions, video_embedings = batch
        fused_emd, soft_prompt = self(frames, video_embedings)  # 只用图像，文本等下处理
        
        noise_std = 0.01
        # 显式 detach，避免多次使用造成 autograd graph 堆积
        fused_feature_1 = fused_emd + torch.randn_like(fused_emd) * noise_std
        fused_feature_2 = fused_emd + torch.randn_like(fused_emd) * noise_std
        soft_prompt_1 = self.model.vis_proj(fused_feature_1)
        soft_prompt_2 = self.model.vis_proj(fused_feature_2)
        infor_loss = simclr_infonce_loss(soft_prompt_1, soft_prompt_2)
        infor_loss_val = infor_loss.item()

        batch_size = frames.shape[0]
        
        """训练过程中的每一步。根据阶段计算不同的损失，并记录日志。"""
        # 第二阶段：计算生成文本与真实文本的交叉熵损失
        torch.cuda.reset_peak_memory_stats()  # 💡 重置显存峰值统计

        # 1. 构造 hard prompt：自然语言提示
    
        # 2. 构造 ground-truth captions
        caption_tokenized = self.tokenizer(captions, padding="max_length", truncation=True, max_length=16, return_tensors="pt")
        
        caption_ids = caption_tokenized.input_ids.to(self.device)
        labels = caption_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100         # pad -> -100
        encoder_outputs = BaseModelOutput(last_hidden_state=soft_prompt)

        
        # 使用hidden state策略
        outputs = self.llm(
            input_ids=caption_ids,
            labels=labels,
            encoder_outputs=encoder_outputs
            # encoder_attention_mask=encoder_attention_mask
        )
        caption_loss = outputs.loss  
        caption_loss_val = caption_loss.item()
        # mlm_weight = 0.1
        # if self.current_epoch < 2:
        #     mlm_loss = compute_mlm_loss(llm=self.llm,tokenizer=self.tokenizer,device=self.device,captions=captions)
        #     mlm_loss_val = mlm_loss.item()
        # else:
        #     mlm_loss_val = 0
         # 使用随 epoch 衰减的 alpha
        # alpha = compute_infor_weight(self.current_epoch, self.trainer.max_epochs, initial_alpha=1.0, final_alpha=0.0)
        # loss = caption_loss + alpha * infor_loss
        
        loss = caption_loss +  0.2 * infor_loss 
       
        total_loss_val = loss.item()
        
        # 5. 计算 loss

        # 记录训练损失
        self.log("train/infor_loss", infor_loss_val, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/caption_loss", caption_loss_val, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train/mlm_loss", mlm_loss_val, on_step=True, on_epoch=True, prog_bar=True)
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
        frames_tensor, caption, video_embedding = batch
        # 1. 生成 caption（预测）
        fused_emd,soft_prompts = self(frames_tensor,video_embedding)
        
        # tem_res = log_per_prompt_statistics(soft_prompts)
        
        # pask key val 策略 1
        generated_captions = self.generate_caption_from_cross_attn(
            soft_prompt=soft_prompts
        )
        
        generated_captions = [strip_prefix(p) for p in generated_captions]
        # 2. 更新 NLP-metrics（BLEU, METEOR, CIDEr 等）
        self.accuracy.update(generated_captions, caption)
        
        if batch_idx == 0 and self.trainer.is_global_zero:
            for i, (pred, ref) in enumerate(zip(generated_captions, caption)):
                if i >= 2:  # 最多记录 2 个样本，防止日志过多
                    break
                self.logger.experiment.add_text(
                    tag=f"val_caption_epoch{self.current_epoch}_sample{i}",
                    text_string=f"**GT:** {ref[:10]}\n**Pred:** {pred}",
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
        """测试过程：生成描述并收集预测和真实值，用于计算指标。"""
        frames_tensor, caption, video_embedding = batch
        # 1. 生成 caption（预测）
        fused_emd,soft_prompts = self(frames_tensor,video_embedding)        
        generated_captions = self.generate_caption_from_cross_attn(
            soft_prompt=soft_prompts
        )
        
        generated_captions = [strip_prefix(p) for p in generated_captions]
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
        # 1. 计算所有 NLP 指标
        score_dict = self.accuracy.compute()

        # 2. 记录每个指标
        # self.log('test/harmonic_mean_bleu_meteor', self.accuracy.harmonice, on_epoch=True, prog_bar=True, logger=True)
        # self.log('test/bleu1', self.accuracy.bleu1, on_epoch=True, prog_bar=True, logger=True)
        # self.log('test/bleu2', self.accuracy.bleu2, on_epoch=True, prog_bar=True, logger=True)
        # self.log('test/bleu3', self.accuracy.bleu3, on_epoch=True, prog_bar=True, logger=True)
        # self.log('test/bleu4', self.accuracy.bleu4, on_epoch=True, prog_bar=True, logger=True)
        # self.log('test/meteor', self.accuracy.meteor, on_epoch=True, prog_bar=True, logger=True)
        # self.log('test/cider', self.accuracy.cider, on_epoch=True, prog_bar=True, logger=True)
        # self.log('test/rougel', self.accuracy.rougel, on_epoch=True, prog_bar=True, logger=True)


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

    
    def configure_optimizers(self):
        """根据阶段返回相应的优化器（和学习率调度器）。"""
            
        # 1. 收集各组参数
        vis_proj_params     = list(self.model.vis_proj.parameters())
        fusion_params = list( self.model.fusion_module.parameters())
        # asy_params = list(self.model.asymmetric_fusion_module.parameters())
        text_enhancer_params = list(self.model.detector.neck.text_enhancer.parameters())
        projoctor_params = list(self.model.projector.parameters())
        # scal_params = list(self.model.scale_attention.parameters())
        # 2. 解冻 LLM 的 decoder 最后2层
        dec_params = []
        total_dec_layers = self.llm.config.decoder_layers
        dec_target_ids = {total_dec_layers - i - 1 for i in range( self.unfreeze_llm_last_n )}
        for name, param in self.llm.model.decoder.named_parameters():
            if any(f"layers.{i}." in name for i in dec_target_ids):
                dec_params.append(param)

        # 3. 定义优化器，分别给不同组不同 lr
        optimizer = torch.optim.AdamW([
            {"params": vis_proj_params, "lr": 5e-5 , "weight_decay": 0.01},
            {"params": projoctor_params, "lr": 7e-5, "weight_decay": 0.01},
            {"params": fusion_params  , "lr": 7e-4, "weight_decay": 0.01},
            {"params": text_enhancer_params, "lr": 5e-4, "weight_decay": 0.01},
            {"params": dec_params, "lr": 9e-6, "weight_decay": 0.01},
          
        ])


        # 4. 学习率调度器（cosine+warmup 举例）
        total_steps = int(self.trainer.estimated_stepping_batches)
        
        scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * total_steps),  # 前10%步数用于warmup
                num_training_steps=int(total_steps),
            )
        
        
        # scheduler = get_polynomial_decay_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=0,  # 也可加一点
        #     num_training_steps=total_steps,
        #     power=0.8  # 线性下降；设成2.0是平方下降
        # )
        
        # return {
        #     "optimizer": optimizer,
            
        # }

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",   # 每step更新
                "frequency": 1,
            },
            "monitor": "val_cider",  # 这行其实可以不加，因为我们用cosine
        }

def compute_infonce_loss_video_level(soft_prompts: torch.Tensor, temperature=0.07):
    """
    soft_prompts: Tensor [B, L, D]，每个视频一个 soft prompt 序列。
    返回：InfoNCE Loss，鼓励不同视频的 soft prompt 差异。
    """
    B, L, D = soft_prompts.size()
    
    # 做 mean pooling 得到 [B, D]
    video_reps = F.normalize(soft_prompts.mean(dim=1), dim=-1)

    # 计算相似度矩阵 [B, B]
    sim_matrix = torch.matmul(video_reps, video_reps.T) / temperature

    # 每一行的对角线是正样本，其他为负样本
    labels = torch.arange(B, device=soft_prompts.device)
    loss = F.cross_entropy(sim_matrix, labels)

    return loss


def simclr_infonce_loss(soft_prompts_1: torch.Tensor, soft_prompts_2: torch.Tensor, temperature=0.07):
    """
    soft_prompts_1 / soft_prompts_2: [B, L, D]，是同一视频的两种视觉视图下生成的 soft prompt
    """
    B, L, D = soft_prompts_1.shape

    # 聚合 soft prompt（可用 mean/pooling）
    z1 = F.normalize(soft_prompts_1.mean(dim=1), dim=-1)  # [B, D]
    z2 = F.normalize(soft_prompts_2.mean(dim=1), dim=-1)  # [B, D]

    # 拼接：共 2B 个样本
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim_matrix = torch.matmul(z, z.T) / temperature  # [2B, 2B]

    # 构造 mask，排除自己（不包括主对角线）
    mask = ~torch.eye(2 * B, dtype=torch.bool, device=z.device)

    # 每个 i 的正样本是 i+B（前半段和后半段对应）
    pos_indices = (torch.arange(B, device=z.device) + B) % (2 * B)
    all_pos_idx = torch.cat([pos_indices, torch.arange(B, device=z.device)])  # [2B]

    # 每个位置的正样本 logit
    logits = sim_matrix[mask].view(2 * B, -1)  # 逐行排除自己 → [2B, 2B-1]
    pos_logit = torch.exp(sim_matrix[torch.arange(2 * B), all_pos_idx])  # [2B]

    denom = torch.sum(torch.exp(logits), dim=1)  # [2B]
    loss = -torch.log(pos_logit / denom)
    return loss.mean()

def compute_infor_weight(current_epoch, total_epochs, initial_alpha=1.0, final_alpha=0.0):
    """线性衰减 infor_loss 权重"""
    progress = min(current_epoch / total_epochs, 1.0)
    return initial_alpha * (1 - progress) + final_alpha * progress


def log_per_prompt_statistics(soft_prompts: torch.Tensor, logger=None, prefix="soft_prompt", step=None):
    """
    记录每个 soft prompt 样本的 mean 和 variance（按 [L, D] 维度进行统计）

    参数:
    - soft_prompts: Tensor [B, L, D]
    - logger: 可选，PyTorch Lightning 的 logger（用于 TensorBoard）
    - prefix: 用于 log 的前缀名
    - step: global_step（可选，控制写入步数）
    """
    B, L, D = soft_prompts.shape

    # 计算每个样本自身的 mean 和 var
    mean_per_prompt = soft_prompts.mean(dim=(1, 2))  # [B]
    var_per_prompt = soft_prompts.var(dim=(1, 2))    # [B]

    if logger is not None:
        for i in range(B):
            logger.experiment.add_scalar(f"{prefix}/sample_{i}_mean", mean_per_prompt[i].item(), global_step=step)
            logger.experiment.add_scalar(f"{prefix}/sample_{i}_variance", var_per_prompt[i].item(), global_step=step)

    return {
        "mean_per_prompt": mean_per_prompt,  # Tensor [B]
        "var_per_prompt": var_per_prompt     # Tensor [B]
    }