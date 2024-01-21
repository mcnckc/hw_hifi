import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
import importlib
from functools import partial
from datetime import datetime

from src.logger import setup_logging
from src.utils import ROOT_PATH


@hydra.main(
    config_path="/home/vladimir/PycharmProjects/NV/src/configs",
    config_name="config.yaml",
    version_base="1.3",
)
class ConfigParser:
    def __init__(self, cfg: DictConfig, resume=None, modification=None, run_id=None):
        self._config = self._update_config(cfg, modification)
        self.resume = cfg.resume
        self.setup_directories(run_id)
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG,
        }

    def _update_config(self, cfg, modification):
        if modification:
            for key, value in modification.items():
                # Добавление или изменение значений в конфигурации
                OmegaConf.update(cfg, key, value)
        return cfg

    def setup_directories(self, run_id):
        save_dir = Path(self._config.trainer.save_dir)
        exper_name = self._config.name
        self._run_id = run_id or datetime.now().strftime(r"%m%d_%H%M%S")
        self._save_dir = save_dir / "models" / exper_name / self._run_id
        self._log_dir = save_dir / "log" / exper_name / self._run_id

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def init_obj(obj_dict, module=None, *args, **kwargs):
        if module is None:
            module = importlib.import_module(obj_dict.module)
        module_name = obj_dict.type
        module_args = dict(obj_dict.args)
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    @staticmethod
    def init_ftn(name, module, *args, **kwargs):
        module_name = name.type
        module_args = dict(name.args)
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def get_logger(self, name, verbosity=2):
        msg_verbosity = "verbosity option {} is invalid. Valid options are {}.".format(
            verbosity,
            self.log_levels.keys(),
        )
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    def save_config(self):
        OmegaConf.save(self._config, self.save_dir / "config.yaml")

    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return Path(self._save_dir)

    @property
    def log_dir(self):
        return Path(self._log_dir)

    @classmethod
    def get_default_configs(cls):
        config_path = ROOT_PATH / "src" / "configs" / "config.json"
        with config_path.open() as f:
            return cls(OmegaConf.load(f))

    @classmethod
    def get_test_configs(cls):
        config_path = ROOT_PATH / "src" / "tests" / "config.json"
        with config_path.open() as f:
            return cls(OmegaConf.load(f))

    def __getitem__(self, name):
        return self._config[name]


if __name__ == "__main__":
    ConfigParser()
