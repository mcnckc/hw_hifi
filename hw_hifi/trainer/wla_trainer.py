import PIL
import torch
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torchvision.transforms import (
    ToTensor,
)
from tqdm import tqdm

from hw_hifi.base import BaseTrainer
from hw_hifi.logger.utils import plot_spectrogram_to_buf
from hw_hifi.utils import (
    inf_loop,
    MetricTracker,
)
from torch.cuda.amp import autocast
from contextlib import contextmanager

from torchaudio.transforms import InverseMelScale
import torchaudio
from hw_hifi.utils.mel import MelSpectrogramConfig

class GriffinLim(torchaudio.transforms.GriffinLim):
    """GriffinLim algorithm as baseline for our task"""

    def __init__(self):
        self.config = MelSpectrogramConfig()
        super().__init__(
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            power=self.config.power,
        )
        self.transform = InverseMelScale(
            sample_rate=self.config.sr,
            n_stft=self.config.n_fft // 2 + 1,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
            norm="slaney",
            mel_scale="slaney",
            n_mels=self.config.n_mels,
        )

    def forward(self, mel_spec):
        return super().forward(self.transform(mel_spec.exp()))
    
@contextmanager
def optional_autocast(enabled=True):
    if enabled:
        with autocast():
            yield
    else:
        yield


class Trainer(BaseTrainer):
    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer_generator,
        optimizer_discriminator,
        config,
        device,
        dataloaders,
        log_predictions_step_epoch=5,
        mixed_precision=False,
        scheduler_generator=None,
        scheduler_discriminator=None,
        len_epoch=None,
        skip_oom=True,
    ):
        super().__init__(
            model,
            criterion,
            metrics=metrics,
            optimizer_g=optimizer_generator,
            optimizer_d=optimizer_discriminator,
            config=config,
            device=device,
            lr_schedulers=[
                scheduler_generator,
                scheduler_discriminator,
            ],
        )
        self.skip_oom = skip_oom
        self.train_dataloader = dataloaders["train"]
        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }
        self.config = config
        if len_epoch is None:
            self.len_epoch = len(self.train_dataloader)
        else:
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.optimizer_generator = optimizer_generator
        self.optimizer_discriminator = optimizer_discriminator
        self.scheduler_generator = scheduler_generator
        self.scheduler_discriminator = scheduler_discriminator
        self.log_predictions_step_epoch = log_predictions_step_epoch
        self.mixed_precision = mixed_precision
        self.train_metrics = MetricTracker(
            *metrics,
            writer=self.writer,
        )
        self.scaler = GradScaler(
            init_scale=512, growth_interval=500, enabled=self.mixed_precision
        )
        self.griffin_lim = GriffinLim().cuda()

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        for tensor_name in ["audio_wave"]:
            print("MOVING TO DEVICE...")
            if tensor_name in batch:
                print("MOVED")
                batch[tensor_name] = batch[tensor_name].to(device)

        return batch

    def _clip_grad_norm(self, optimizer):
        self.scaler.unscale_(optimizer)
        if self.config["trainer"].get("grad_norm_clip") is not None:
            try:
                clip_grad_value_(
                    parameters=self.model.parameters(),
                    clip_value=self.config["trainer"]["grad_max_abs"],
                )
                clip_grad_norm_(
                    parameters=self.model.parameters(),
                    max_norm=self.config["trainer"]["grad_norm_clip"],
                    error_if_nonfinite=True,
                )
            except RuntimeError:
                return False
        return True

    def _train_epoch(self, epoch):
        self.model.train()
        self.criterion.train()
        self.train_metrics.reset()
        for i, batch in enumerate(
            tqdm(
                self.train_dataloader,
                desc="train",
                total=self.len_epoch,
            )
        ):
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            if i == 0:
                last_train_metrics = self.debug(
                    batch,
                    i,
                    epoch,
                )
            elif i >= self.len_epoch:
                break
        self.scheduler_generator.step()
        self.scheduler_discriminator.step()
        log = last_train_metrics
        for part, dataloader in self.evaluation_dataloaders.items():
            self._evaluation_epoch(epoch, part, dataloader)
        return log

    @torch.no_grad()
    def debug(self, batch, batch_idx, epoch):
        self.logger.debug(
            "Train Epoch: {} {} Loss D/G: {:.4f}/{:.4f}".format(
                epoch,
                self._progress(batch_idx),
                self.train_metrics.avg("discriminator_loss"),
                self.train_metrics.avg("generator_loss"),
            )
        )
        self._log_scalars(self.train_metrics)
        last_train_metrics = self.train_metrics.result()
        self.train_metrics.reset()
        if self.writer is not None:
            self._log_spectrogram(batch, mode="train")
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.writer.add_scalar(
                "epoch",
                epoch,
            )
            self.writer.add_scalar(
                "learning rate generator",
                self.optimizer_generator.state_dict()["param_groups"][0]["lr"],
            )
            self.writer.add_scalar("scaler factor", self.scaler.get_scale())
            self.writer.add_scalar(
                "learning rate discriminator",
                self.optimizer_discriminator.state_dict()["param_groups"][0]["lr"],
            )
            audio_generator_example = (
                batch["wave_fake_detached"][0]
                .detach()
                .cpu()
                .to(torch.float32)
                .numpy()
                .flatten()
            )
            audio_true_example = (
                batch["wave_true"][0].detach().cpu().to(torch.float32).numpy().flatten()
            )
            self.writer.add_audio(
                "generated",
                audio_generator_example,
                sample_rate=22050,
            )
            self.writer.add_audio(
                "true",
                audio_true_example,
                sample_rate=22050,
            )
            self.writer.add_audio(
                f"griffin-lim",
                self.griffin_lim(batch["mel_true"][0].detach()).cpu().flatten(),
                sample_rate=22050,
            )
        return last_train_metrics

    def process_batch(
        self,
        batch,
        metrics: MetricTracker,
    ):
        batch = self.move_batch_to_device(batch, self.device)
        with torch.no_grad():
            batch.update(self.model(**batch))
        with optional_autocast(enabled=self.mixed_precision):
            batch.update(self.model.generator(**batch))

            # Discriminator
            self.optimizer_discriminator.zero_grad(set_to_none=True)

            batch.update(self.model.mp_discriminator(**batch, detach=True))
            batch.update(self.model.ms_discriminator(**batch, detach=True))
            batch.update(self.criterion.discriminator_loss(**batch))

        self.scaler.scale(batch["discriminator_loss"]).backward()
        if not self._clip_grad_norm(self.optimizer_discriminator):
            print("NaN gradients. Skipping batch")
            self.scaler.update()
            return batch
        self.train_metrics.update(
            "grad_norm_discriminator",
            self.get_grad_norm(),
        )
        self.scaler.step(self.optimizer_discriminator)
        self.scaler.update()
        with optional_autocast(enabled=self.mixed_precision):
            # Generator
            self.optimizer_generator.zero_grad(set_to_none=True)
            batch.update(
                self.model.mp_discriminator(
                    **batch,
                    detach=False,
                )
            )
            batch.update(
                self.model.ms_discriminator(
                    **batch,
                    detach=False,
                )
            )

            batch.update(self.criterion.generator_loss(**batch))
        self.scaler.scale(batch["generator_loss"]).backward()
        if not self._clip_grad_norm(self.optimizer_generator):
            print("NaN gradients. Skipping batch")
            self.scaler.update()
            return batch
        self.scaler.step(self.optimizer_generator)
        self.scaler.update()
        self.train_metrics.update(
            "grad_norm_generator",
            self.get_grad_norm(),
        )

        for item in batch:
            if item in self.train_metrics.keys():
                metrics.update(
                    item,
                    batch[item].item(),
                )
        return batch

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(
            self.train_dataloader,
            "n_samples",
        ):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(
            current,
            total,
            100.0 * current / total,
        )

    @torch.no_grad()
    def _evaluation_epoch(self, epoch, part, dataloader):
        if self.writer is None:
            return
        self.model.eval()
        for batch_idx, batch in tqdm(
            enumerate(dataloader),
            desc=part,
            total=len(dataloader),
        ):
            batch = self.move_batch_to_device(batch, self.device)
            batch.update(self.model(**batch))
            batch.update(self.model.generator(**batch))

            self.writer.add_audio(
                f"test_generated_{batch_idx}",
                batch["wave_fake"].cpu().flatten(),
                sample_rate=22050,
                caption=f"#{batch_idx}",
            )
            self._log_spectrogram(batch, mode="test", idx=str(batch_idx))

            if epoch == 1:
                self.writer.add_audio(
                    f"test_true_{batch_idx}",
                    batch["wave_true"].cpu().flatten(),
                    sample_rate=22050,
                    caption=f"True #{batch_idx}",
                )
                self.writer.add_audio(
                    f"test_griffinlim_{batch_idx}",
                    self.griffin_lim(batch["mel_true"]).cpu().flatten(),
                    sample_rate=22050,
                    caption=f"GriffinLim #{batch_idx}",
                )
        return

    @staticmethod
    def make_image(buff):
        return ToTensor()(PIL.Image.open(buff))

    @torch.no_grad()
    def _log_spectrogram(self, batch, mode="train", idx=""):
        spectrogram_types = [
            "_true",
            "_fake",
        ]
        for spectrogram_type in spectrogram_types:
            spectrogram = (
                batch[f"mel{spectrogram_type}"][0].detach().cpu().to(torch.float64)
            )
            spectrogram = torch.nan_to_num(spectrogram)
            buf = plot_spectrogram_to_buf(spectrogram)
            self.writer.add_image(
                f"{mode}{idx}_mel_{spectrogram_type}",
                Trainer.make_image(buf),
            )

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [
                    torch.norm(
                        torch.nan_to_num(
                            p.grad,
                            nan=0,
                        ).detach(),
                        norm_type,
                    ).cpu()
                    for p in parameters
                ]
            ),
            norm_type,
        )
        return total_norm.item()

    @torch.no_grad()
    def _log_scalars(
        self,
        metric_tracker: MetricTracker,
    ):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(
                metric_name,
                metric_tracker.avg(metric_name),
            )