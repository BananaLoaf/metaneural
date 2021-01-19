PACKAGE_NAME = "metaneural"
__version__ = "1.0"


from typing import Type

import tensorflow as tf

from metaneural.config import *
from metaneural.runner import DefaultRunner


def train(config_class: Type[DefaultConfig], runner_class: Type[DefaultRunner], description: str = "Metaneural"):
    gpu_devs = [dev.name for dev in tf.config.list_logical_devices("GPU") + tf.config.list_logical_devices("XLA_GPU")]
    cpu_devs = [dev.name for dev in tf.config.list_logical_devices("CPU") + tf.config.list_logical_devices("XLA_CPU")]
    config_class.devices[KWARGS][HELP] = config_class.devices[KWARGS][HELP].format(gpu_devs, cpu_devs)

    config = config_class.cli(description)
    runner_class.new_run(config)


def resume(config_class: Type[DefaultConfig], runner_class: Type[DefaultRunner], description: str = "Metaneural"):
    resume_config = ResumeConfig.cli(description)

    if resume_config.checkpoint_epoch is None:
        print("Loading latest checkpoint")
        checkpoint_path = sorted(Path(resume_config.path, "checkpoint").glob("*"), key=lambda p: int(p.name), reverse=True)[0]
    else:
        print(f"Loading epoch {resume_config.checkpoint_epoch} checkpoint")
        checkpoint_path = Path(resume_config.path, "checkpoint").glob(f"*{str(resume_config.checkpoint_epoch)}").__next__()

    config = config_class.load(checkpoint_path / "config.yaml")
    runner_class.resume(config=config,
                        run_directory=Path(resume_config.path),
                        checkpoint_path=checkpoint_path)


def convert(config_class: Type[DefaultConfig], runner_class: Type[DefaultRunner], description: str = "Metaneural"):
    converter_config = ConverterConfig.cli(description)

    if converter_config.checkpoint_epoch is None:
        print("Loading latest checkpoint")
        checkpoint_path = sorted(Path(converter_config.path, "checkpoint").glob("*"), key=lambda p: p.stat().st_mtime_ns)[-1]
    else:
        print(f"Loading checkpoint {converter_config.checkpoint_epoch}")
        checkpoint_path = Path(converter_config.path, "checkpoint") / str(converter_config.checkpoint_epoch)

    config = config_class.load(checkpoint_path / "config.yaml")
    runner_class.convert(config=config, converter_config=converter_config,
                         run_directory=Path(converter_config.path), checkpoint_path=checkpoint_path)
