from pprint import pprint
import argparse
import os
from datetime import datetime
import random
import numpy as np
import gc
from tqdm import tqdm
import torch
import glob
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from cifake_dataset import CIFAKEDataset
from coco_fake_dataset import COCOFakeDataset
from dffd_dataset import DFFDDataset
import main_model as model
from lib.util import load_config
import os
from lightning.pytorch.loggers import CSVLogger
import os
import lightning as L
from fvcore.nn import FlopCountAnalysis,parameter_count


class SaveSpecificEpochsCallback(L.pytorch.callbacks.Callback):
    def __init__(self, target_epochs, save_dir, filename_prefix):
        super().__init__()
        # 确保传入的是一个列表，例如 [5, 6, 7]
        self.target_epochs = target_epochs
        self.save_dir = save_dir
        self.filename_prefix = filename_prefix

        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        # current_epoch 从 0 开始，所以人类理解的第 N 轮是 current_epoch + 1
        current = trainer.current_epoch + 1

        # 如果当前轮次在目标列表中，就保存
        if current in self.target_epochs:
            # 构造文件名，建议加上指标占位符或者直接写死
            # 这里为了简单，直接用 epoch 编号
            filename = f"{self.filename_prefix}_epoch{current}.ckpt"
            ckpt_path = os.path.join(self.save_dir, filename)

            trainer.save_checkpoint(ckpt_path)
            print(f"[Info] 已保存第 {current} 轮的权重: {ckpt_path}")


def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        help="The path to the config.",
        default="./configs/results_cifake_T.cfg",
    )
    args = parser.parse_args()
    return args


class TqdmTrainCallback(L.pytorch.callbacks.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            total = sum(trainer.num_training_batches) if isinstance(trainer.num_training_batches,
                                                                    list) else trainer.num_training_batches
            self.pbar = tqdm(
                total=total,
                desc=f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}",
                unit="batch"
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_rank == 0:
            self.pbar.update(1)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            self.pbar.close()
def Flops(model):
  with torch.no_grad():
    x = torch.randn(1, 3, 224, 224).to("cuda")
    flops = FlopCountAnalysis(model, x)
    total_flops = flops.total()
    total_params = parameter_count(model)[""]
  # 打印到控制台
  print(f"\n[INFO] Model FLOPs: {total_flops / 1e9:.6f}G")
  print(f"[INFO] Model Parameters: {total_params / 1e6:.3f}M\n")


if __name__ == "__main__":
    seed= [5,35,42]
    for seed in seed:
        gc.collect()
        torch.cuda.empty_cache()
        args = args_func()
        cfg = load_config(args.cfg)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.set_float32_matmul_precision("medium")
        print(cfg["dataset"]["name"])
        data = cfg["dataset"]["name"]
        # get data
        if data == "coco_fake":
            print(
                f"Loading COCO-Fake datasets from {cfg['dataset']['coco2014_path']} and {cfg['dataset']['coco_fake_path']}"
            )
            train_dataset = CIFAKEDataset(
                dataset_path=cfg["dataset"]["cifake_path"],
                split="train",
                resolution=cfg["train"]["resolution"],
            )
            val_dataset = COCOFakeDataset(
                coco2014_path=cfg["dataset"]["coco2014_path"],
                coco_fake_path=cfg["dataset"]["coco_fake_path"],
                split="val",
                mode="single",
                resolution=cfg["train"]["resolution"],
            )
        elif data == "dffd":
            print(f"Loading DFFD dataset from {cfg['dataset']['dffd_path']}")
            train_dataset = DFFDDataset(
                dataset_path=cfg["dataset"]["dffd_path"],
                split="train",
                resolution=cfg["train"]["resolution"],
            )
            val_dataset = DFFDDataset(
                dataset_path=cfg["dataset"]["dffd_path"],
                split="test",
                resolution=cfg["train"]["resolution"],
            )
        elif data == "cifake":
            print(f"Loading CIFAKE dataset from {cfg['dataset']['cifake_path']}")
            train_dataset = CIFAKEDataset(
                dataset_path=cfg["dataset"]["cifake_path"],
                split="train",
                resolution=cfg["train"]["resolution"],
            )
            val_dataset = DFFDDataset(
                dataset_path=cfg["dataset"]["dffd_path"],
                split="test",
                resolution=cfg["train"]["resolution"],
            )
            # val_dataset = COCOFakeDataset(
            #     coco2014_path=cfg["dataset"]["coco2014_path"],
            #     coco_fake_path=cfg["dataset"]["coco_fake_path"],
            #     split="val",
            #     mode="single",
            #     resolution=cfg["train"]["resolution"],
            # )
            # val_dataset = COCOFakeDataset(
            #     coco2014_path=cfg["dataset"]["coco2014_path"],
            #     coco_fake_path=cfg["dataset"]["coco_fake_path"],
            #     split="val",
            #     mode="single",
            #     resolution=cfg["train"]["resolution"],
            # )
            # val_dataset = CIFAKEDataset(
            #     dataset_path=cfg["dataset"]["cifake_path"],
            #     split="test",
            #     resolution=cfg["train"]["resolution"],
            # )
        # loads the dataloaders
        num_workers = 4
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["train"]["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # init model
        positive_samples = sum([item["is_real"] for item in train_dataset.items])
        negative_samples = len(train_dataset) - positive_samples
        #ckpt_dir = fr"/root/OurCode/lpb_only_seed_{seed}"
        #ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
        #net = model.BNext4DFR.load_from_checkpoint(ckpt_files[0])
        net = model.BNext4DFR(
            num_classes=cfg["dataset"]["labels"],
            backbone=cfg["model"]["backbone"],
            freeze_backbone=cfg["model"]["freeze_backbone"],
            add_magnitude_channel=cfg["model"]["add_magnitude_channel"],
            add_fft_channel=cfg["model"]["add_fft_channel"],
            add_lbp_channel=cfg["model"]["add_lbp_channel"],
            pos_weight=negative_samples / positive_samples,
            use_vib=False,
            use_fuse=False,
            use_rgbfreq=True,
            use_CBNN=True,
            use_dyvib=False,
            learning_rate=5e-5,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = net.to(device)
        NAME = "OAM+CBNEXT"
        specific_epochs_callback = SaveSpecificEpochsCallback(
            target_epochs=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],  # <--- 这里指定你想保存的任意轮次
            save_dir=f"{data}_{NAME}_seed_{seed}",  # 保存路径
            filename_prefix=f"{NAME}_{data}_{cfg['dataset']['name']}_{cfg['model']['backbone']}"
        )
        # start training
        date = datetime.now().strftime("%Y%m%d_%H%M")
        project = "DFAD_CVPRW24"
        run_label = args.cfg.split("/")[-1].split(".")[0]
        run = NAME + data + f"{seed}"
        logger = CSVLogger("logs", name=run)
        trainer = L.Trainer(
            accelerator="gpu",
            devices=[0],
            #strategy="ddp_find_unused_parameters_true",
            precision="16-mixed" if cfg["train"]["mixed_precision"] else 32,
            gradient_clip_algorithm="norm",
            gradient_clip_val=1.0,
            accumulate_grad_batches=cfg["train"]["accumulation_batches"],
            limit_train_batches=cfg["train"]["limit_train_batches"],
            limit_val_batches=cfg["train"]["limit_val_batches"],
            max_epochs=cfg["train"]["epoch_num"],
            # num_sanity_val_steps=-1,
            val_check_interval=1.0,
            check_val_every_n_epoch=1,
            callbacks=[
                L.pytorch.callbacks.ModelCheckpoint(
                    save_top_k=2,
                    mode="max",
                    dirpath="/checkpoints",
                    filename= NAME + data + "_" + cfg["model"][
                        "backbone"] + "_{epoch}-{train_auc:.10f}-{val_acc:.10f}",
                ),
                specific_epochs_callback,
                TqdmTrainCallback(),

            ],
            # callbacks=[specific_epochs_callback],
            logger=logger,
        )
        # val_results = trainer.validate(model=net, dataloaders=val_loader)
        # print("REAULT DETAIL：", val_results)
        trainer.fit(model=net, train_dataloaders=train_loader, val_dataloaders=val_loader)
