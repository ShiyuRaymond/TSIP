import torch
import torch.nn.functional as F
from transformers import DataCollatorForLanguageModeling    

def compute_mlm_loss(llm, tokenizer, device, captions, mlm_probability: float = 0.15):
    encoder = llm.model.encoder  # Bart encoder

    # Tokenize captions
    tokenized = tokenizer(
        captions,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    # 使用 DataCollator 自动构造 masked input 和 labels
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability
    )
    tokenized_samples = [tokenizer(c, truncation=True, padding="max_length", max_length=32) for c in captions]

    batch = collator(tokenized_samples)  # 👈 正确形式是 List[Dict[str, List[int]]]
    batch = {k: torch.tensor(v).to(device) for k, v in batch.items()}


    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    # Forward through encoder
    encoder_outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)

    # 投影回 vocab 空间
    hidden = encoder_outputs.last_hidden_state  # [B, T, 768]
    logits = llm.lm_head(hidden)           # [B, T, vocab_size]

    # CrossEntropyLoss
    mlm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
    return mlm_loss


import spacy

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_svo_triplet(sentence):
    doc = nlp(sentence)
    subj, verb, obj = None, None, None
    for token in doc:
        if "subj" in token.dep_:
            subj = token.text
        if token.pos_ == "VERB":
            verb = token.text
        if "obj" in token.dep_:
            obj = token.text
    return subj, verb, obj


def compute_svo_loss(refined_prompt, decoder_embeds, caption_text, tokenizer):
    losses = []
    for i, caption in enumerate(caption_text):
        subj, verb, obj = extract_svo_triplet(caption)
        if not subj or not verb or not obj:
            continue

        tokenized = tokenizer(caption, padding="max_length", truncation=True, max_length=refined_prompt.shape[1], return_tensors="pt")
        input_ids = tokenized.input_ids[0]  # [L]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        def find_token_idx(word):
            for idx, tok in enumerate(tokens):
                if word.lower() in tok.lower():
                    return idx
            return None

        idx_subj = find_token_idx(subj)
        idx_verb = find_token_idx(verb)
        idx_obj  = find_token_idx(obj)

        max_len = decoder_embeds.shape[1]
        if None in [idx_subj, idx_verb, idx_obj] or max(idx_subj, idx_verb, idx_obj) >= max_len:
            continue

        loss = (
            F.mse_loss(refined_prompt[i, 0], decoder_embeds[i, idx_subj]) +
            F.mse_loss(refined_prompt[i, 1], decoder_embeds[i, idx_verb]) +
            F.mse_loss(refined_prompt[i, 2], decoder_embeds[i, idx_obj])
        )
        losses.append(loss)

    if losses:
        return torch.stack(losses).mean()
    else:
        return torch.tensor(0.0, requires_grad=True)
    
    
def compute_align_loss(refined_prompt: torch.Tensor, decoder_input_embeds: torch.Tensor):
    """
    Computes MSE alignment loss between soft prompt and decoder input embeddings.
    """
    N = min(refined_prompt.shape[1], decoder_input_embeds.shape[1])
    loss = F.mse_loss(refined_prompt[:, :N, :], decoder_input_embeds[:, :N, :])
    return loss



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
    
    
def scheduled_sampling(inputs, outputs, step, total_steps):
    import random
    prob = min(1.0, step / total_steps)
    new_inputs = inputs.clone()
    for i in range(1, inputs.size(1)):
        if random.random() < prob:
            new_inputs[:, i] = outputs[:, i-1].argmax(dim=-1)
    return new_inputs

def embedding_align_loss(soft_prompt, caption_embeds):
   
    N = min(soft_prompt.shape[1], caption_embeds.shape[1])
    loss = F.mse_loss(soft_prompt[:, :N, :], caption_embeds[:, :N, :])
    return loss


def mlm_loss(captions, tokenizer, llm):
    inputs = tokenizer(captions, return_tensors='pt', padding=True, truncation=True, max_length=16).to(llm.device)
    labels = inputs.input_ids.clone()
    # 随机mask掉15%的token
    probability_matrix = torch.full(labels.shape, 0.15).to(labels.device)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100
    inputs.input_ids[masked_indices] = tokenizer.mask_token_id

    outputs = llm(**inputs, labels=labels)
    return outputs.loss


def no_repeat_ngram_loss(pred_ids, n=2):
    """
    pred_ids: Tensor, [B, T], 解码器预测的token id序列
    n: n-gram的n
    返回：每个batch的no_repeat_ngram损失
    """
    loss = 0
    batch_size, seq_len = pred_ids.size()
    for i in range(batch_size):
        seq = pred_ids[i].tolist()
        ngram_set = set()
        for j in range(seq_len - n + 1):
            ngram = tuple(seq[j:j+n])
            if ngram in ngram_set:
                loss += 1
            else:
                ngram_set.add(ngram)
    # 归一化为batch平均
    loss = loss / batch_size
    return loss


def ending_ngram_match_loss(pred_ids, gt_ids, n=4, eos_token_id=2, pad_token_id=1):
    """
    pred_ids, gt_ids: [B, T], token id
    n: ngram长度
    eos_token_id: <eos>的id
    pad_token_id: <pad>的id
    """
    batch_size, seq_len = pred_ids.size()
    loss = 0
    for i in range(batch_size):
        pred = pred_ids[i].tolist()
        gt = gt_ids[i].tolist()
        # 只保留eos前内容
        if eos_token_id in pred:
            pred = pred[:pred.index(eos_token_id)]
        if eos_token_id in gt:
            gt = gt[:gt.index(eos_token_id)]
        # 去掉pad
        pred = [t for t in pred if t != pad_token_id]
        gt = [t for t in gt if t != pad_token_id]
        if len(pred) < n or len(gt) < n:
            loss += 1
        else:
            if pred[-n:] != gt[-n:]:
                loss += 1
    loss = loss / batch_size
    return loss

import spacy
nlp = spacy.load("en_core_web_sm")

def grammar_error_loss(sentences):
    # 返回非完整句子的比例
    error_cnt = 0
    for s in sentences:
        doc = nlp(s)
        if not any([sent.root.pos_ == 'VERB' for sent in doc.sents]):
            error_cnt += 1
    return error_cnt / len(sentences)




class SoftPromptCenterLoss(nn.Module):
    def __init__(self, mode="cosine", alpha=1.0, beta=0.0):
        """
        mode: "mse", "cosine", or "combine"
        alpha, beta: 仅combine模式下权重
        """
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss(reduction='batchmean')
    def forward(self, soft_prompt, center_tensor):
        """
        soft_prompt: [B, D] 或 [D]
        center_tensor: [B, D] 或 [D]
        """
        if soft_prompt.dim() == 1:
            soft_prompt = soft_prompt.unsqueeze(0)
        if center_tensor.dim() == 1:
            center_tensor = center_tensor.unsqueeze(0)
        
        if self.mode == "mse":
            loss = self.mse(soft_prompt, center_tensor)
        elif self.mode == "cosine":
            # CosineSimilarity: 返回值越大越相似，所以用 1-相似度作为loss
            cos = F.cosine_similarity(soft_prompt, center_tensor, dim=-1)
            loss = 1 - cos.mean()
        elif self.mode == "combine":
            mse_loss = self.mse(soft_prompt, center_tensor)
            cos = F.cosine_similarity(soft_prompt, center_tensor, dim=-1)
            cos_loss = 1 - cos.mean()
            loss = self.alpha * mse_loss + self.beta * cos_loss
        elif self.mode =='KL':
            log_soft_prompt = F.log_softmax(soft_prompt, dim=-1)
            center_prob = F.softmax(center_tensor, dim=-1)
            loss = self.kl(log_soft_prompt, center_prob)
        else:
            raise ValueError("mode must be mse, cosine or combine")
        return loss
    


import torch
import torch.nn.functional as F

def soft_prompt_cosine_diversity_loss(soft_prompts):
    """
    soft_prompts: [B, L, D]  # B=batch size, L=token数, D=特征维
    返回: 多样性正则loss，值越小表示越不多样
    """
    B, L, D = soft_prompts.shape
    # 先对每个soft prompt token取平均，得到 [B, D]
    avg_prompt = soft_prompts.mean(dim=1)  # [B, D]
    # 归一化
    avg_prompt = F.normalize(avg_prompt, dim=-1)
    # 计算 batch 内所有两两余弦相似度
    sim_matrix = avg_prompt @ avg_prompt.T   # [B, B]
    # 去除对角线（自己和自己相似度为1）
    mask = ~torch.eye(B, dtype=torch.bool, device=soft_prompts.device)
    diversity_loss = sim_matrix[mask].mean()
    return diversity_loss
