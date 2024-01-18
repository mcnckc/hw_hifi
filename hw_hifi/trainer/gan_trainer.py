import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from hw_hifi.base import BaseTrainer
from hw_hifi.base.base_text_encoder import BaseTextEncoder
from hw_hifi.logger.utils import plot_spectrogram_to_buf
from hw_hifi.utils import inf_loop, MetricTracker


class GanTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            metrics,
            optimizer_g,
            optimizer_d,
            config,
            device,
            dataloaders,
            lr_scheduler_g=None,
            lr_scheduler_d=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, None, metrics, optimizer_g, optimizer_d, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            print('HERE')
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            print('There')
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        print('Train Length', self.len_epoch)
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler_g = lr_scheduler_g
        self.lr_scheduler_d = lr_scheduler_d
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "loss", "grad norm", "discriminator loss", "mel loss",
            "feature loss",
            "adversarial loss",
            "total generator loss", *[m.name for m in self.metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", "discriminator loss", "mel loss",
            "feature loss",
            "adversarial loss",
            "total generator loss", *[m.name for m in self.metrics], writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["spectrogram", "audio_wave"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc='Train epoch', total=self.len_epoch)
        ):
            if batch_idx >= self.len_epoch:
                break
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                print("OOM")
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                """
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                """
                self.writer.add_scalar(
                    "discriminator learning rate", self.optimizer_d.param_groups[0]['lr']
                )
                self.writer.add_scalar(
                    "generator learning rate", self.optimizer_g.param_groups[0]['lr']
                )
                #self._log_predictions(**batch)
                #self._log_spectrogram(batch["spectrogram"])
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
        
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer_d.zero_grad()

        true = batch['audio_wave'].unsqueeze(dim=1)
        fake = self.model.generator(batch['spectrogram'])[..., :true.shape[-1]]
        fake_spec = self.model.mel(fake)
        
        true_out_mp, fake_out_mp, _, _ = self.model.mp_discriminator(true, fake.detach())
        true_out_ms, fake_out_ms, _, _ = self.model.ms_discriminator(true, fake.detach())

        d_loss = self.model.discr_loss(true_out_mp, fake_out_mp) + self.model.discr_loss(true_out_ms, fake_out_ms)
        if is_train:
            d_loss.backward()
            self.optimizer_d.step()
        
        if is_train:
            self.optimizer_g.zero_grad()
        
        loss_mel = F.l1_loss(batch['spectrogram'], fake_spec)
        
        true_out_mp, fake_out_mp, true_fs_mp, fake_fs_mp = self.model.mp_discriminator(true, fake)
        true_out_ms, fake_out_ms, true_fs_ms, fake_fs_ms = self.model.ms_discriminator(true, fake)

        loss_feature = self.model.feature_loss(true_fs_mp, fake_fs_mp) + self.model.feature_loss(true_fs_ms, fake_fs_ms)
        loss_gen = self.model.generator_loss(fake_out_mp) + self.model.generator_loss(fake_out_ms)
        total_gen_loss = loss_mel * 45 + loss_feature * 2 + loss_gen
        
        if is_train:
            loss_mel.backward()
            self.optimizer_g.step()
        
        
        batch.update({'audio_wave': fake, 
                'd_loss': d_loss, 
                'mel_loss': loss_mel * 45,
                'feature_loss': loss_feature * 2,
                'gen_loss': loss_gen,
                'total_gen_loss': total_gen_loss})
        
        if is_train:
            if self.lr_scheduler_g is not None:
                if isinstance(self.lr_scheduler_g, ReduceLROnPlateau):
                    self.lr_scheduler_g.step(batch['loss'].item())
                else:
                    self.lr_scheduler_g.step()
            
            if self.lr_scheduler_d is not None:
                if isinstance(self.lr_scheduler_d, ReduceLROnPlateau):
                    self.lr_scheduler_d.step(batch['loss'].item())
                else:
                    self.lr_scheduler_d.step()

        metrics.update("discriminator loss", batch["d_loss"].item())
        metrics.update("mel loss", batch["mel_loss"].item())
        metrics.update("feature loss", batch["feature_loss"].item())
        metrics.update("adversarial loss", batch["gen_loss"].item())
        metrics.update("total generator loss", batch["total_gen_loss"].item())
        for met in self.metrics:
            metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
