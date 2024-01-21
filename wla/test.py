import os
import warnings

import hydra
import numpy as np
import torch
import torchaudio

import src.model as module_model
from src.trainer import Trainer
from src.utils import ROOT_PATH
from src.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore")

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


@hydra.main(config_path="src/configs", config_name="config.yaml")
def main(config):
    config = ConfigParser(config)
    logger = config.get_logger("test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    print(config["resume"])
    checkpoint = torch.load(config["resume"], map_location=device)
    print("Checkpoint!")
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    args = config["test_settings"]
    output_dir = args.out_dir
    with torch.no_grad():
        if args.mel_dir is not None:
            print(f"Processing mel files from {args.mel_dir}")
            for mel_file in filter(
                lambda f: f.endswith(".npy"), os.listdir(args.mel_dir)
            ):
                mel_path = os.path.join(args.mel_dir, mel_file)
                mel_data = np.load(mel_path)
                mel_tensor = torch.tensor(mel_data).to(device)
                generated_audio = (
                    model.generator(mel_tensor)["wave_fake"].cpu().view(1, -1)
                )
                output_filename = os.path.splitext(mel_file)[0] + ".wav"
                torchaudio.save(
                    os.path.join(args.out_dir, output_filename),
                    generated_audio,
                    sample_rate=args.sample_rate,
                    format="wav",
                )
        elif args.audio_dir is not None:
            print(f"Processing audios from {args.audio_dir}")
            os.makedirs(os.path.join(output_dir), exist_ok=True)
            for i, audio_file in enumerate(
                list(filter(lambda f: f.endswith(".wav"), os.listdir(args.audio_dir)))
            ):
                batch = {
                    "waves": torchaudio.load(os.path.join(args.audio_dir, audio_file))[
                        0
                    ]
                }
                batch = Trainer.move_batch_to_device(batch, device)
                batch.update(model(**batch))
                batch.update(model.generator(**batch))
                generated_audio = batch["wave_fake"].cpu().view(1, -1)
                output_filename = os.path.splitext(audio_file)[0] + ".wav"
                torchaudio.save(
                    os.path.join(args.out_dir, output_filename),
                    generated_audio,
                    sample_rate=args.sample_rate,
                    format="wav",
                )


if __name__ == "__main__":
    main()
