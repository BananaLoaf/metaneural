PACKAGE_NAME = "metaneural"
__version__ = "1.0"


from metaneural.config import *
from metaneural.runner import DefaultRunner


def train(config_type: DefaultConfig, runner_type: DefaultRunner, description: str = "Metaneural"):
    gpu_devs = [dev.name for dev in tf.config.list_logical_devices("GPU") + tf.config.list_logical_devices("XLA_GPU")]
    cpu_devs = [dev.name for dev in tf.config.list_logical_devices("CPU") + tf.config.list_logical_devices("XLA_CPU")]
    config_type.devices[KWARGS][HELP] = config_type.devices[KWARGS][HELP].format(gpu_devs, cpu_devs)

    config = config_type.cli(description)
    runner_type.new_run(config)
