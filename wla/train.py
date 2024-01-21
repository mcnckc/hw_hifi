import warnings
import numpy as np
import torch

import src.loss as module_loss
import src.model as module_arch
from src.trainer import Trainer
from src.utils import prepare_device
from src.utils.object_loading import get_dataloaders
from src.utils.parse_config import ConfigParser
import random
import hydra

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)


@hydra.main(config_path="src/configs", config_name="config.yaml")
def main(cfg):
    config = ConfigParser(cfg)
    logger = config.get_logger("train")
    # torch.autograd.set_detect_anomaly(True)
    print(
        f"Running training.\n"
        f"Deterministic: {torch.are_deterministic_algorithms_enabled()}"
    )
    dataloaders = get_dataloaders(config)
    model = config.init_obj(config["arch"], module_arch)
    logger.info(model)
    device, device_ids = prepare_device(config["n_gpu"])
    print(device, device_ids)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    loss_module = config.init_obj(config["loss"], module_loss).to(device)
    metrics = config["metrics"]
    optimizer_generator = config.init_obj(
        config["optimizer"],
        torch.optim,
        filter(lambda p: p.requires_grad, model.generator.parameters()),
    )
    optimizer_discriminator = config.init_obj(
        config["optimizer"],
        torch.optim,
        list(filter(lambda p: p.requires_grad, model.mp_discriminator.parameters()))
        + list(filter(lambda p: p.requires_grad, model.ms_discriminator.parameters())),
    )
    scheduler_generator = config.init_obj(
        config["lr_scheduler"], torch.optim.lr_scheduler, optimizer_generator
    )
    scheduler_discriminator = config.init_obj(
        config["lr_scheduler"], torch.optim.lr_scheduler, optimizer_discriminator
    )
    print(
        "Num params discriminator MP:",
        sum(p.numel() for p in model.mp_discriminator.parameters() if p.requires_grad),
    )
    print(
        "Num params discriminator MS:",
        sum(p.numel() for p in model.ms_discriminator.parameters() if p.requires_grad),
    )
    print(
        "Num params generator:",
        sum(p.numel() for p in model.generator.parameters() if p.requires_grad),
    )
    print("Num params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    trainer = Trainer(
        model,
        loss_module,
        metrics,
        optimizer_generator,
        optimizer_discriminator,
        config=config,
        device=device,
        dataloaders=dataloaders,
        scheduler_generator=scheduler_generator,
        scheduler_discriminator=scheduler_discriminator,
        len_epoch=config["trainer"].get("len_epoch", None),
        log_predictions_step_epoch=config["trainer"].get(
            "log_predictions_step_epoch", 1
        ),
        mixed_precision=config["trainer"].get("mixed_precision", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
