# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from MSVD_MODE.TSIP import  TSIP
# from MSR_VTT_MODE.encoder_decoder import  TSIP

from datamodules.videoclip_dataloader import MSRVTTDataModule


from omegaconf import OmegaConf
import gc
from pytorch_lightning import seed_everything

seed_everything(11, workers=True)


def load_partial_state_dict(model, state_dict):
    model_dict = model.state_dict()
    filtered_dict = {}

    for k, v in state_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                filtered_dict[k] = v
            else:
                print(f"Skipping: {k} due to shape mismatch: "
                      f"model {model_dict[k].shape} vs checkpoint {v.shape}")
        else:
            print(f"Skipping: {k} not found in model.")

    # 加载过滤后的参数
    model.load_state_dict(filtered_dict, strict=False)



def train(best_ckpt_stage1=None, unfreeze= 3):
    # 加载配置
    data_cfg = OmegaConf.load("configs/data.yaml")
    stage2_cfg = OmegaConf.load('configs/stage2.yaml')
    stage2_cfg.base_model = OmegaConf.load('configs/base_model.yaml')

    # 加载数据
    data_module = MSRVTTDataModule(
        train_json=data_cfg.train_json,
        val_json=data_cfg.val_json,
        test_json=data_cfg.test_json,
        video_dir=data_cfg.video_dir,
        feature_dir=data_cfg.feature_dir,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        max_caption_length=data_cfg.max_caption_length,
        eval_cfg= data_cfg.eval_cfg,
        frames_dir=data_cfg.frames_dir
    )

    # Logger & Callbacks
    logger = TensorBoardLogger(
        save_dir=stage2_cfg.trainer.log_dir,
        name=stage2_cfg.trainer.logger_name
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        monitor='val/rougel',
        mode='max',
        save_top_k=1,
        filename='stage2-best-{epoch:02d}-{val_cider:.3f}',
        save_last=True
    )
    early_stop_callback = EarlyStopping(
        # monitor='val/cider',
        monitor='val/rougel',
        patience=20,
        mode='max'
    )

    # 模型初始化
    model = TSIP(
        stage2_cfg,
        lr=stage2_cfg.model.lr,
        dropout=stage2_cfg.model.dropout,
        grad_clip=stage2_cfg.model.grad_clip,
    )

    # 加载 stage 1 的 checkpoint 权重（部分加载）
    if best_ckpt_stage1 is not None:
        checkpoint = torch.load(best_ckpt_stage1, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)



    # 训练器初始化
    trainer = pl.Trainer(
        strategy="ddp_find_unused_parameters_true",
        accelerator='gpu',
        devices=stage2_cfg.trainer.gpus,
        max_epochs=stage2_cfg.trainer.max_epochs,
        gradient_clip_val=stage2_cfg.trainer.gradient_clip,
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
        logger=logger,
        precision=stage2_cfg.trainer.precision,
        log_every_n_steps=100
    )

    trainer.fit(model, datamodule=data_module)

    print("Training complete.")
    return checkpoint_callback.best_model_path, stage2_cfg, data_module



def only_test(checkpoint_path):
     # 加载配置
    data_cfg = OmegaConf.load("configs/data.yaml")
    stage2_cfg = OmegaConf.load('configs/stage2.yaml')
    stage2_cfg.base_model = OmegaConf.load('configs/base_model.yaml')

    # 加载数据
    # data_module = MSRVTTDataModule(
    #     train_json=data_cfg.train_json,
    #     val_json=data_cfg.val_json,
    #     test_json=data_cfg.test_json,
    #     video_dir=data_cfg.video_dir,
    #     feature_dir=data_cfg.feature_dir,
    #     batch_size=data_cfg.batch_size,
    #     num_workers=data_cfg.num_workers,
    #     max_caption_length=data_cfg.max_caption_length,
    #     eval_cfg= data_cfg.eval_cfg,
    #     frames_dir=data_cfg.frames_dir
    # )
    data_module = MSRVTTDataModule(
        train_json=data_cfg.train_json,
        val_json=data_cfg.val_json,
        test_json=data_cfg.test_json,
        video_dir=data_cfg.video_dir,
        feature_dir=data_cfg.feature_dir,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        max_caption_length=data_cfg.max_caption_length,
    )

    test(checkpoint_path, stage2_cfg, data_module)
    

def test(checkpoint_path, stage2_cfg, data_module):
    print(f"Testing using checkpoint: {checkpoint_path}")
    
    # 释放训练显存
    torch.cuda.empty_cache()
    gc.collect()

    # 重新初始化模型
    model = TSIP.load_from_checkpoint(
        checkpoint_path,
        stage=stage2_cfg.stage,
        llm_name=stage2_cfg.model.llm_name,
        strict=False
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=stage2_cfg.trainer.gpus,
        precision=stage2_cfg.trainer.precision
    )

    data_module.setup()
    results = trainer.test(model, dataloaders=data_module.test_dataloader())
    print("Test results:", results)


def exp(best_ckpt_stage1=None):
    ckpt_path, stage2_cfg, data_module = train(best_ckpt_stage1)
    test(ckpt_path, stage2_cfg, data_module)


def exp8(best_ckpt_stage1):
    ckpt_path, stage2_cfg, data_module = train(best_ckpt_stage1)
    test(ckpt_path, stage2_cfg, data_module)

def training(best_ckpt_stage1,max_exp =  6):
    ckpt_path = best_ckpt_stage1
    for unfreeze in range(3, max_exp+1):
        ckpt_path, stage2_cfg, data_module = train(ckpt_path, unfreeze)
    pass 

def main():
    # only_test('/workspace/YOLO-World/scripts/yoloworld_lightning/logs/msrvtt/stage2/version_190/checkpoints/stage2-best-epoch=01-val_cider=0.000.ckpt')
    # exp('/workspace/YOLO-World/scripts/yoloworld_lightning/logs/msrvtt/stage2/version_162/checkpoints/stage2-best-epoch=00-val_cider=0.000.ckpt')
    only_test('/workspace/YOLO-World/scripts/yoloworld_lightning/logs/exp5/exp5_v2/checkpoints/stage2-best-epoch=00-val_cider=0.000.ckpt')
if __name__ == "__main__":
    main()

