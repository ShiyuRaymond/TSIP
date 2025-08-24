import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import json
import os
import cv2
from torchvision import transforms
import numpy as np
from transformers import PreTrainedTokenizer

import random

class MSRVTTDataset(Dataset):
    def __init__(self, json_path, video_dir, feature_dir, max_caption_length=32, transform=None, stage='train'):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.video_dir = video_dir
        # self.tokenizer = tokenizer
        self.max_caption_length = max_caption_length
        self.feature_dir = feature_dir
        self.stage = stage
        
        self.transform = transform if transform else transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        video_id = item['video_id']
        if self.stage == 'train':
            caption = random.choice(item['caption'])
        else:
            caption =item['caption']
            
        video_path = os.path.join(self.video_dir, video_id + '.avi')
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_sampled_frames = self.max_caption_length

        if total_frames <= 0:
            raise ValueError(f"Video {video_path} has no frames.")

        if total_frames >= num_sampled_frames:
            indices = np.linspace(0, total_frames - 1, num=num_sampled_frames, dtype=int)
        else:
            indices = np.arange(total_frames)
            indices = np.pad(indices, (0, num_sampled_frames - total_frames), mode='wrap')

        frames = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                transformed = self.transform(frame_rgb)
                frames.append(transformed)
        cap.release()
        

        if len(frames) < num_sampled_frames:
            repeat = num_sampled_frames - len(frames)
            frames.extend(frames[:repeat])
        


        frames_tensor = torch.stack(frames[:num_sampled_frames])  # [32, 3, 224, 224]
        

        video_embedding = torch.load(os.path.join(self.feature_dir, video_id + '.pt'))
        
        return frames_tensor, caption, video_embedding


class MSRVTTDataModule(pl.LightningDataModule):
    def __init__(self, train_json, val_json, test_json, video_dir, feature_dir,
                 batch_size=8, num_workers=8, max_caption_length=32):
        super().__init__()
        self.train_json = train_json
        self.val_json = val_json
        self.test_json = test_json
        self.video_dir = video_dir
        self.feature_dir = feature_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_caption_length = max_caption_length

    def setup(self, stage=None):
        self.train_dataset = MSRVTTDataset(self.train_json, self.video_dir,self.feature_dir,
                                            self.max_caption_length,stage='train')
        self.val_dataset = MSRVTTDataset(self.val_json, self.video_dir,self.feature_dir,
                                          self.max_caption_length,stage='val')
        self.test_dataset = MSRVTTDataset(self.test_json, self.video_dir,self.feature_dir,
                                          self.max_caption_length,stage='test')

    def collate_fn(self, batch):
        # video_paths, frames, input_ids, attention_masks, video_embedings = zip(*batch)
        frames, captions, video_embedings = zip(*batch)
        
        frames = torch.stack(frames)
        # input_ids = torch.stack(input_ids)
        # attention_masks = torch.stack(attention_masks)
        
        video_embedings = torch.stack(video_embedings)
        # return video_paths, frames, input_ids, attention_masks, video_embedings
        return frames, captions, video_embedings

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)
