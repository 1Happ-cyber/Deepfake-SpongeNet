from pprint import pprint
import argparse
import gc
from os.path import join
from datetime import datetime
import os
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from cifake_dataset import CIFAKEDataset
from lightning.pytorch.loggers import CSVLogger
from coco_fake_dataset import COCOFakeDataset
from dffd_dataset import DFFDDataset
import main_model as model
from lib.util import load_config
import random
import numpy as np
from tqdm import tqdm


class TqdmProgressCallback(L.pytorch.callbacks.Callback):
    def on_test_start(self, trainer, pl_module):
        self.pbar = tqdm(total=trainer.num_test_batches[0], desc="Testing")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.pbar.update(1)

    def on_test_end(self, trainer, pl_module):
        self.pbar.close()

def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        help="The path to the config.",
        default=r"./configs/ablation_baseline.cfg",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    
    args = args_func()

    # load configs
    cfg = load_config(args.cfg)
    pprint(cfg)

    # preliminary setup
    torch.manual_seed(cfg["test"]["seed"])
    random.seed(cfg["test"]["seed"])
    np.random.seed(cfg["test"]["seed"])
    torch.set_float32_matmul_precision("medium")

    # get data
    if cfg["dataset"]["name"] == "coco_fake":
        print(
            f"Load COCO-Fake datasets from {cfg['dataset']['coco2014_path']} and {cfg['dataset']['coco_fake_path']}"
        )
        test_dataset = COCOFakeDataset(
            coco2014_path=cfg["dataset"]["coco2014_path"],
            coco_fake_path=cfg["dataset"]["coco_fake_path"],
            split="val",
            mode="single",
            resolution=cfg["test"]["resolution"],
        )
    elif cfg["dataset"]["name"] == "dffd":
        print(f"Load DFFD dataset from {cfg['dataset']['dffd_path']}")
        test_dataset = DFFDDataset(
            dataset_path=cfg["dataset"]["dffd_path"],
            split="test",
            resolution=cfg["test"]["resolution"],
        )
    elif cfg["dataset"]["name"] == "cifake":
        print(f"Loading CIFAKE dataset from {cfg['dataset']['cifake_path']}")
        test_dataset = CIFAKEDataset(
            dataset_path=cfg["dataset"]["cifake_path"],
            split="test",
            resolution=cfg["test"]["resolution"],
        )

    # loads the dataloaders
    num_workers = 4
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["test"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    net = model.BNext4DFR.load_from_checkpoint(cfg["test"]["weights_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    # start training
    date = datetime.now().strftime("%Y%m%d_%H%M")
    project = "ck25"
    run_label = args.cfg.split("/")[-1].split(".")[0]
    run = cfg["dataset"]["name"] + f"_test_{date}_{run_label}"


    logger = CSVLogger("logs", name=run)
    trainer = L.Trainer(
        accelerator="gpu" if "cuda" in str(device) else "cpu",
        devices=4,
        strategy="ddp_find_unused_parameters_true",
        precision="16-mixed" if cfg["test"]["mixed_precision"] else 32,
        limit_test_batches=cfg["test"]["limit_test_batches"],
        logger=logger,
        enable_progress_bar=True,
        callbacks=[TqdmProgressCallback()],
    )
    trainer.test(model=net, dataloaders=test_loader)
