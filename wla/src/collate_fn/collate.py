import torch
from torch.nn.utils.rnn import (
    pad_sequence,
)


def collate_fn(batch):
    waves = pad_sequence(
        [torch.from_numpy(item["wave"]) for item in batch],
        batch_first=True,
        padding_value=0,
    )
    return {"waves": waves}
