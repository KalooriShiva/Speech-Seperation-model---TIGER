import os
import sys
import torch
from torch import Tensor
import argparse
import json
import yaml
import look2hear.datas
import look2hear.models
import look2hear.losses
import look2hear.metrics
from glob import glob
import torchaudio
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from rich.console import Console
from pytorch_lightning.loggers import WandbLogger
from rich import print
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

import wandb
# wandb.login()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--conf_dir",
    default="Experiments/checkpoint/TIGER-VoiceBank/conf.yml",
    help="Full path to configuration file",
)

class VoiceBankDemandDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', sample_rate=16000):
        self.root_dir = root_dir
        self.split = split
        self.sample_rate = sample_rate
        self.noisy_dir = os.path.join(root_dir, f"noisy_{split}set_wav")
        self.clean_dir = os.path.join(root_dir, f"clean_{split}set_wav")
        self.noisy_files = sorted(glob(os.path.join(self.noisy_dir, "*.wav")))
        if not self.noisy_files:
            raise ValueError(f"No WAV files found in {self.noisy_dir}")
        self.clean_files = [f.replace("noisy", "clean") for f in self.noisy_files]
        for clean_file in self.clean_files:
            if not os.path.exists(clean_file):
                raise ValueError(f"Clean file not found: {clean_file}")

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_path = self.noisy_files[idx]
        clean_path = self.clean_files[idx]
        noisy, sr = torchaudio.load(noisy_path)
        clean, _ = torchaudio.load(clean_path)
        if sr != self.sample_rate:
            noisy = torchaudio.functional.resample(noisy, sr, self.sample_rate)
            clean = torchaudio.functional.resample(clean, sr, self.sample_rate)
        if noisy.shape[0] > 1:
            noisy = torch.mean(noisy, dim=0, keepdim=True)
        if clean.shape[0] > 1:
            clean = torch.mean(clean, dim=0, keepdim=True)
        return noisy, clean

def collate_fn(batch, max_len_samples=32000, normalize_audio=False):
    """Collate function to pad/truncate audio pairs to a fixed length."""
    noisy_list, clean_list = zip(*batch)
    noisy_padded = []
    clean_padded = []
    for n, c in zip(noisy_list, clean_list):
        if n.shape[-1] > max_len_samples:
            n = n[..., :max_len_samples]
        else:
            pad_amount = max_len_samples - n.shape[-1]
            n = torch.nn.functional.pad(n, (0, pad_amount))
        if c.shape[-1] > max_len_samples:
            c = c[..., :max_len_samples]
        else:
            pad_amount = max_len_samples - c.shape[-1]
            c = torch.nn.functional.pad(c, (0, pad_amount))
        if normalize_audio:
            n = n / (n.abs().max() + 1e-8)
            c = c / (c.abs().max() + 1e-8)
        noisy_padded.append(n)
        clean_padded.append(c)
    return torch.stack(noisy_padded), torch.stack(clean_padded)

class VoiceBankDemandDataModule(pl.LightningDataModule):
    def __init__(self, data_config):
        super().__init__()
        self.data_config = data_config

    def setup(self, stage=None):
        root_dir = self.data_config.get("dataset_path", "/home/kaloori.shiva/Thesis/TIGER/Voicebank_DEMAND")
        self.train_dataset = VoiceBankDemandDataset(root_dir, split="train", sample_rate=self.data_config["sample_rate"])
        self.val_dataset = VoiceBankDemandDataset(root_dir, split="test", sample_rate=self.data_config["sample_rate"])
        self.test_dataset = VoiceBankDemandDataset(root_dir, split="test", sample_rate=self.data_config["sample_rate"])
        # Debug: Print sample file paths
        if stage == "fit":
            print(f"Sample train files: {self.train_dataset.noisy_files[:2]}")
            print(f"Sample val files: {self.val_dataset.noisy_files[:2]}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.data_config["batch_size"],
            shuffle=True,
            num_workers=self.data_config.get("num_workers", 4),
            pin_memory=True,
            collate_fn=lambda batch: collate_fn(
                batch,
                max_len_samples=int(2.0 * self.data_config["sample_rate"]),
                normalize_audio=self.data_config.get("normalize_audio", False)
            )
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_config["batch_size"],
            shuffle=False,
            num_workers=self.data_config.get("num_workers", 4),
            pin_memory=True,
            collate_fn=lambda batch: collate_fn(
                batch,
                max_len_samples=int(2.0 * self.data_config["sample_rate"]),
                normalize_audio=self.data_config.get("normalize_audio", False)
            )
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.data_config["batch_size"],
            shuffle=False,
            num_workers=self.data_config.get("num_workers", 4),
            pin_memory=True,
            collate_fn=lambda batch: collate_fn(
                batch,
                max_len_samples=int(2.0 * self.data_config["sample_rate"]),
                normalize_audio=self.data_config.get("normalize_audio", False)
            )
        )

    @property
    def make_loader(self):
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()

class SeparationSystem(pl.LightningModule):
    def __init__(self, audio_model, loss_func, optimizer, train_loader, val_loader, test_loader, scheduler, config):
        super().__init__()
        self.audio_model = audio_model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.config = config
        self.accumulate_grad_batches = config.get("training", {}).get("accumulate_grad_batches", 1)

    def forward(self, x):
        return self.audio_model(x)

    def training_step(self, batch, batch_idx):
        noisy, clean = batch
        # Log input statistics
        self.log("noisy_mean", noisy.mean(), on_step=True, on_epoch=False)
        self.log("noisy_std", noisy.std(), on_step=True, on_epoch=False)
        self.log("clean_mean", clean.mean(), on_step=True, on_epoch=False)
        self.log("clean_std", clean.std(), on_step=True, on_epoch=False)
        
        est_clean = self(noisy)
        loss = self.loss_func["train"](est_clean.squeeze(1), clean.squeeze(1)).mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Log gradient norm
        grad_norm = 0
        for param in self.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        self.log("grad_norm", grad_norm, on_step=True, on_epoch=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
        noisy, clean = batch
        est_clean = self(noisy)
        loss = self.loss_func["val"](est_clean.squeeze(1), clean.squeeze(1)).mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # Debug: Print tensor statistics for first batch
        if batch_idx == 0:
            print(f"Val batch {batch_idx}: est_clean mean={est_clean.mean().item():.4f}, std={est_clean.std().item():.4f}")
            print(f"Val batch {batch_idx}: clean mean={clean.mean().item():.4f}, std={clean.std().item():.4f}")
        return loss

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": "val_loss",
            },
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        if (batch_idx + 1) % self.accumulate_grad_batches == 0 or (batch_idx + 1) == self.trainer.num_training_batches:
            optimizer.step(closure=optimizer_closure)
            optimizer.zero_grad()
            # Log learning rate
            lr = self.optimizer.param_groups[0]["lr"]
            self.log("learning_rate", lr, on_step=True, on_epoch=False)
        else:
            optimizer_closure()

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

def main(config):
    print("Instantiating datamodule <VoiceBankDemandDataModule>")
    datamodule = VoiceBankDemandDataModule(config["datamodule"]["data_config"])
    datamodule.setup()

    train_loader, val_loader, test_loader = datamodule.make_loader
    
    print("Instantiating AudioNet <{}>".format(config["audionet"]["audionet_name"]))
    model = getattr(look2hear.models, config["audionet"]["audionet_name"])(
        sample_rate=config["datamodule"]["data_config"]["sample_rate"],
        **config["audionet"]["audionet_config"],
    )
    
    print("Instantiating Optimizer <Adam>")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["lr"])
    
    print("Instantiating Scheduler <ReduceLROnPlateau>")
    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        verbose=True
    )
    
    config["main_args"]["exp_dir"] = os.path.join(
        os.getcwd(), "Experiments", "checkpoint", config["exp"]["exp_name"]
    )
    exp_dir = config["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(config, outfile)

    print("Instantiating Loss <SingleSrcNegSDR>")
    loss_func = {
        "train": getattr(look2hear.losses, "SingleSrcNegSDR")(
            sdr_type="sisdr",
            **config["loss"]["train"]["config"],
        ),
        "val": getattr(look2hear.losses, "SingleSrcNegSDR")(
            sdr_type="sisdr",
            **config["loss"]["val"]["config"],
        ),
    }

    print("Instantiating System <SeparationSystem>")
    system = SeparationSystem(
        audio_model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scheduler=scheduler,
        config=config,
    )

    print("Instantiating Callbacks")
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir)
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        filename="{epoch}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        verbose=True,
        save_last=True,
    )
    callbacks.append(checkpoint)
    callbacks.append(EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=20,
        verbose=True
    ))
    callbacks.append(RichProgressBar())

    gpus = config["training"]["gpus"] if torch.cuda.is_available() else None
    distributed_backend = "cuda" if torch.cuda.is_available() else None

    logger_dir = os.path.join(os.getcwd(), "Experiments", "tensorboard_logs")
    os.makedirs(os.path.join(logger_dir, config["exp"]["exp_name"]), exist_ok=True)
    comet_logger = WandbLogger(
        name=config["exp"]["exp_name"],
        save_dir=os.path.join(logger_dir, config["exp"]["exp_name"]),
        project="Real-work-dataset",
        offline=True
    )

    trainer = pl.Trainer(
        max_epochs=500,
        callbacks=callbacks,
        default_root_dir=exp_dir,
        devices=[0],
        accelerator="cuda",
        strategy="auto",
        limit_train_batches=1.0,
        gradient_clip_val=5.0,
        logger=comet_logger,
        sync_batchnorm=False,
        accumulate_grad_batches=config["training"].get("accumulate_grad_batches", 1),
        precision="16-mixed",
    )
    trainer.fit(system)
    print("Finished Training")
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.audio_model.serialize()
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.conf_dir) as f:
        config = yaml.safe_load(f)
    main(config)