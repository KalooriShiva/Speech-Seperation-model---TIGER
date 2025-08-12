import os
import argparse
import yaml
import torch
import look2hear.models
import look2hear.datas
from look2hear.metrics import MetricsTracker
from look2hear.utils import tensors_to_device, RichProgressBarTheme, MyMetricsTextColumn, BatchesProcessedColumn
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--conf_dir",
    default="Experiments/checkpoint/TIGER-EchoSet/conf.yml",
    help="Full path to the config YAML file"
)
parser.add_argument(
    "--checkpoint",
    default="epoch=110.ckpt",
    help="Checkpoint file to load (e.g., epoch=9.ckpt or last.ckpt)"
)
parser.add_argument(
    "--test_subset",
    type=int,
    default=100,
    help="Number of test samples to process (default: 100, use 0 for full set)"
)

def main(config, checkpoint, test_subset):
    metricscolumn = MyMetricsTextColumn(style=RichProgressBarTheme.metrics)
    progress = Progress(
        TextColumn("[bold blue]Testing", justify="right"),
        BarColumn(bar_width=None),
        "•",
        BatchesProcessedColumn(style=RichProgressBarTheme.batch_progress),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        "•",
        metricscolumn
    )

    # Construct experiment directory and checkpoint path
    exp_dir = os.path.join(
        os.getcwd(), "Experiments", "checkpoint", config["exp"]["exp_name"]
    )
    model_path = os.path.join(exp_dir, checkpoint)
    print(f"Loading model from checkpoint: {model_path}")

    # Instantiate model
    model_class = getattr(look2hear.models, config["audionet"]["audionet_name"])
    model = model_class(
        sample_rate=config["datamodule"]["data_config"]["sample_rate"],
        **config["audionet"]["audionet_config"]
    )

    # Load checkpoint and strip 'audio_model.' prefix
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get('state_dict', checkpoint)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("audio_model."):
            new_state_dict[k[len("audio_model."):]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    # Device setup
    if "gpus" in config.get("training", {}) and config["training"]["gpus"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()

    # Prepare datamodule
    datamodule_class = getattr(look2hear.datas, config["datamodule"]["data_name"])
    datamodule = datamodule_class(**config["datamodule"]["data_config"])
    datamodule.setup()
    _, _, test_set = datamodule.make_sets
    print(f"Test dataset size: {len(test_set)}")

    # Limit test set if subset is specified
    test_size = len(test_set) if test_subset == 0 else min(test_subset, len(test_set))
    print(f"Testing on {test_size} samples")

    # Results save directory
    ex_save_dir = os.path.join(exp_dir, "results")
    os.makedirs(ex_save_dir, exist_ok=True)

    metrics = MetricsTracker(save_file=os.path.join(ex_save_dir, "metrics.csv"))
    print(f"Saving metrics to: {os.path.join(ex_save_dir, 'metrics.csv')}")

    torch.no_grad().__enter__()
    with progress:
        for idx in progress.track(range(test_size)):
            mix, sources, key = tensors_to_device(test_set[idx], device=device)
            est_sources = model(mix[None])

            # Convert tensors to CPU numpy for metrics or saving
            est_sources_np = est_sources.squeeze(0).cpu()
            mix_np = mix.cpu()
            sources_np = sources.cpu()

            # Compute metrics
            metrics(mix=mix_np, clean=sources_np, estimate=est_sources_np, key=key)
            print(f"Processed idx {idx}, metrics updated")  # Debug metrics

            # Save mix, s1, s2 files
            save_dir = os.path.join(ex_save_dir, f"idx{idx}")
            os.makedirs(save_dir, exist_ok=True)
            filename = key.split("/")[-1]
            import torchaudio
            # Save mix
            torchaudio.save(os.path.join(save_dir, f"mix.wav"), mix_np.unsqueeze(0), 16000)
            # Save s1, s2
            for i in range(est_sources_np.shape[0]):
                filepath = os.path.join(save_dir, f"s{i+1}.wav")
                torchaudio.save(filepath, est_sources_np[i].unsqueeze(0), 16000)

    # Force metrics save
    metrics.final()
    # metrics.writer.
    print(f"Metrics finalized and saved to: {metrics}")

if __name__ == "__main__":
    args = parser.parse_args()

    # Load config YAML
    with open(args.conf_dir, "r") as f:
        train_conf = yaml.safe_load(f)

    main(train_conf, args.checkpoint, args.test_subset)