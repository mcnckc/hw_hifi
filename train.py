import argparse
import collections
import warnings

import numpy as np
import torch

import hw_hifi.loss as module_loss
import hw_hifi.metric as module_metric
import hw_hifi.model as module_arch
from hw_hifi.trainer.gan_trainer import GanTrainer
from hw_hifi.utils import prepare_device
from hw_hifi.utils.object_loading import get_dataloaders
from hw_hifi.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    print('Cuda', torch.cuda.is_available())
    logger = config.get_logger("train")


    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture, then print to console
    model = config.init_obj(config["arch"], module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.num_params()
    # get function handles of loss and metrics
    #loss_module = config.init_obj(config["loss"], module_loss).to(device)
    metrics = [
        config.init_obj(metric_dict, module_metric)
        for metric_dict in config["metrics"]
    ]

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer_g = config.init_obj(config["optimizer_g"], torch.optim, model.generator_params())
    optimizer_d = config.init_obj(config["optimizer_d"], torch.optim, model.discriminator_params())
    lr_scheduler_g = config.init_obj(config["lr_scheduler_g"], torch.optim.lr_scheduler, optimizer_g)
    lr_scheduler_d = config.init_obj(config["lr_scheduler_d"], torch.optim.lr_scheduler, optimizer_d)

    trainer = GanTrainer(
        model,
        metrics,
        optimizer_g,
        optimizer_d,
        config=config,
        device=device,
        dataloaders=dataloaders,
        lr_scheduler_g=lr_scheduler_g,
        lr_scheduler_d=lr_scheduler_d,
        len_epoch=config["trainer"].get("len_epoch", None)
    )
    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
