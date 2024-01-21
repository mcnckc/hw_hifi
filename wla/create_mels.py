import os
import argparse
import numpy as np
import torchaudio
from src.model.mel_spectrogram import MelSpectrogramConfig, MelSpectrogram


def main(args):
    mel_creator = MelSpectrogram(MelSpectrogramConfig())
    os.makedirs(args.out_dir, exist_ok=True)
    for wave_file in filter(lambda x: x.endswith('.wav'), os.listdir(args.wav_dir)):
        audio, sr = torchaudio.load(args.wav_dir + '/' + wave_file)
        mel = mel_creator(audio).cpu().numpy()
        np.save(args.out_dir + '/' + wave_file[:-4] + '.npy', mel)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "--wav_dir",
        default='test_data',
        type=str,
        help="Where are waves",
    )
    args.add_argument(
        "--out_dir",
        default='mel_test_data',
        type=str,
        help="Where to store mels",
    )
    main(args.parse_args())