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