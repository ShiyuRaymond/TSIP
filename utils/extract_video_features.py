from yolo_world.models.backbones.hugging_videoclip_backbone import HuggingVideoCLIPBackbone



import os
import torch
from glob import glob
from tqdm import tqdm

def process_single_video_embeddings(video_dir, save_dir, video_backbone, exts=['.mp4', '.avi', '.mov']):
    """
    对每个视频提取特征并保存为同名 .pt 文件

    Args:
        video_dir (str): 包含视频的文件夹
        save_dir (str): 保存 .pt 文件的文件夹
        video_backbone (callable): 接收 List[str]（长度为1）并返回 Tensor 的函数
        exts (list[str]): 支持的视频扩展名
    """
    os.makedirs(save_dir, exist_ok=True)

    video_files = [f for f in sorted(glob(os.path.join(video_dir, '*')))
                   if os.path.splitext(f)[1].lower() in exts]

    print(f"Found {len(video_files)} videos in {video_dir}")

    for video_path in tqdm(video_files, desc="Processing videos"):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        save_path = os.path.join(save_dir, f"{video_name}.pt")

        try:
            # video_backbone 要求返回 Tensor 或 [Tensor]
            embedding = video_backbone([video_path])
            if isinstance(embedding, list) and len(embedding) == 1:
                embedding = embedding[0]
            torch.save(embedding.cpu(), save_path)
        except Exception as e:
            print(f"[ERROR] Failed to process {video_path}: {e}")


def main():
    eval_config = 'video_clip_v0.1.yaml'
    device = 'cuda'

    video_backbone = HuggingVideoCLIPBackbone(eval_config=eval_config, projector_dim=512, device=device)
    
    process_single_video_embeddings(
    video_dir='data/msrvtt/videos/TrainValVideo',
    save_dir='data/msrvtt/features',
    video_backbone=video_backbone
)
    
if __name__ == '__main__':
    
    main()