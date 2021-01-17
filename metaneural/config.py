from typing import Tuple, Any, Callable
from argparse import ArgumentParser
import yaml

from pathlib import Path


ARGS = "ARGS"
KWARGS = "KWARGS"
GROUP_NAME = "GROUP_NAME"
# EXCLUSIVE_GROUP = "EXCLUSIVE_GROUP"
CONSTANT = "CONSTANT"
SAVE = "SAVE"

TYPE = "type"
ACTION = "action"
NARGS = "nargs"
REQUIRED = "required"
DEFAULT = "default"
CHOICES = "choices"
HELP = "help"


class ConfigBuilder:
    def __init__(self):
        self.__schemes__ = {}

        # Move schemes to __schemes__ and set all fields to None
        for field in self.get_fields():
            scheme = getattr(self, field)

            setattr(self, field, None)
            self.__schemes__[field] = self.set_defaults(scheme)

    @staticmethod
    def set_defaults(scheme: dict) -> dict:
        """Set all default values for a scheme"""
        scheme.setdefault(ARGS, [])
        scheme.setdefault(KWARGS, {})
        scheme.setdefault(SAVE, True)
        return scheme

    def get_fields(self) -> str:
        """Fetches all fields"""
        all_vars = {**vars(self.__class__)}
        for cls in self.__class__.__bases__:
            if cls.__name__ == ConfigBuilder.__name__:
                break
            else:
                all_vars = {**all_vars, **vars(cls)}

        for attr, value in all_vars.items():
            if not (attr.startswith("__") and attr.endswith("__")) and not isinstance(value, Callable):
                yield attr

    def get_field_scheme_value(self) -> Tuple[str, dict, Any]:
        """Fetches all available fields, schemes and values"""
        for field in self.get_fields():
            yield field, getattr(self.__class__, field), self.__schemes__[field]

    ################################################################
    def save(self, path: Path):
        """Dump yaml representation into the file"""
        self._cleanup()
        with path.open("w") as file:
            yaml.dump(self.to_dict(), file, indent=4)

    @classmethod
    def load(cls, path: Path):
        """Load config from yaml file"""
        self = cls()
        self._cleanup()

        with path.open("r") as file:
            data = yaml.load(file)

        for field, in self.get_fields():
            try:
                setattr(self, field, data[field])
            except KeyError:
                raise KeyError(f"Config is missing required key '{field}'")

        return self

    def to_dict(self) -> dict:
        data = {}
        for field, scheme, value in self.get_field_scheme_value():
            data[field] = value
        return data

    def __repr__(self):
        return self.to_dict()

    ################################################################
    @classmethod
    def cli(cls, description: str):
        """Creates command line arguments parser"""
        self = cls()

        ################################################################
        # Create parser
        parser = ArgumentParser(description=description)
        groups = {}

        for field, scheme, value in self.get_field_scheme_value():
            # Set constants and skip
            if CONSTANT in scheme.keys():
                setattr(self, field, scheme[CONSTANT])
                continue

            # Create group and set as target for new argument
            if GROUP_NAME in scheme.keys():
                group_name = scheme[GROUP_NAME]

                groups.setdefault(group_name, parser.add_argument_group(group_name))
                target_parser = groups[group_name]

            else:
                target_parser = parser

            try:
                if scheme[ARGS][0].startswith("-"):
                    scheme[KWARGS]["dest"] = field
            except IndexError:
                pass
            target_parser.add_argument(*scheme[ARGS], **scheme[KWARGS])

        ################################################################
        # Parse
        args = parser.parse_args()
        for field, value in vars(args).items():
            setattr(self, field, value)

        return self

    def _cleanup(self):
        to_remove = []
        for field, scheme, value in self.get_field_scheme_value():
            if not scheme[SAVE]:
                to_remove.append(field)

        for field in to_remove:
            delattr(self, field)
            delattr(self.__class__, field)
            del self.__schemes__[field]


class DefaultConfig(ConfigBuilder):
    """
    This config implementation allows to easily trace param usage with the help of IDE

    Examples:
    name = {GROUP_NAME: "Model",                                       # Not required
            ARGS: ["--name"],
            KWARGS: {TYPE: str, REQUIRED: True, HELP: "Model name"}},
            SAVE: False                                                # If save in json, (default: True)

    # Does not provide cli param, just exists, can be saved
    step = {CONSTANT: 0,
            SAVE; False}

    # Not used
    device = {GROUP_NAME: "Device params",
              EXCLUSIVE_GROUP: [
                  {ARGS: ["--cpu"],
                   KWARGS: {TYPE: str, DEFAULT: "/device:CPU:0", CHOICES: [dev.name for dev in tf.config.list_logical_devices("CPU")], HELP: "CPU (default: %(default)s)"}},
                  {ARGS: ["--gpu"],
                   KWARGS: {TYPE: str, HELP: "GPUs"}}
              ],
              REQUIRED: False}  # Only used with EXCLUSIVE_GROUP, if not required, one of elements in a group must have DEFAULT value (default: True)
    """

    name = {ARGS: ["--name"],
            KWARGS: {TYPE: str, REQUIRED: True, HELP: "Model name"}}

    ################################################################
    # Device params
    use_tpu = {GROUP_NAME: "Device",
               ARGS: ["--use-tpu"],
               KWARGS: {ACTION: "store_true",
                        HELP: "Use Google Cloud TPU. If True, --dev is ignored (default: %(default)s)"}}
    tpu_name = {GROUP_NAME: "Device",
                ARGS: ["--tpu-name"],
                KWARGS: {TYPE: str,
                         DEFAULT: None,
                         HELP: "Google Cloud TPU name, if None and flag --use-tpu is set, will try to detect automatically (default: %(default)s)"}}
    devices = {GROUP_NAME: "Device",
               ARGS: ["--dev"],
               KWARGS: {NARGS: "+",
                        TYPE: str,
                        DEFAULT: ["/device:CPU:0", ],
                        HELP: "Available GPUs: {}, available CPUs: {}"}}

    ################################################################
    # Training params
    epoch = {CONSTANT: 0}
    epochs = {GROUP_NAME: "Training",
              ARGS: ["-e", "--epochs"],
              KWARGS: {TYPE: int,
                      REQUIRED: True,
                      HELP: "Epochs (default: %(default)s)"}}
    steps_per_epoch = {GROUP_NAME: "Training",
                       ARGS: ["-se", "--steps-per-epoch"],
                       KWARGS: {TYPE: int,
                                DEFAULT: None,
                                HELP: "Steps per epoch, if None the epoch will run until the train dataset is exhausted (default: %(default)s)"}}
    ################################################################
    test_split = {GROUP_NAME: "Training",
                  ARGS: ["-ts"],
                  KWARGS: {TYPE: float,
                           DEFAULT: 0.1,
                           HELP: "Test split (default: %(default)s)"}}
    test_steps = {GROUP_NAME: "Training",
                  ARGS: ["-ts", "--test-steps"],
                  KWARGS: {TYPE: int,
                           DEFAULT: None,
                           HELP: "Test steps per test, if None the epoch will run until the test dataset is exhausted (default: %(default)s)"}}
    ################################################################
    batch_size = {GROUP_NAME: "Training",
                  ARGS: ["-b", "--batch-size"],
                  KWARGS: {TYPE: int,
                           DEFAULT: 1,
                           HELP: "Batch size (default: %(default)s)"}}
    q_aware_train = {GROUP_NAME: "Training",
                     ARGS: ["-qat", "--quantization-aware-training"],
                     KWARGS: {NARGS: "+",
                              TYPE: int,
                              DEFAULT: [0],
                              HELP: "Quantization aware training for chosen models, https://www.tensorflow.org/model_optimization/guide/quantization/training (default: %(default)s)"}}

    ################################################################
    # Other
    checkpoint_freq = {GROUP_NAME: "Other",
                       ARGS: ["-cf", "--checkpoint-freq"],
                       KWARGS: {TYPE: int,
                                REQUIRED: True,
                                HELP: "Checkpoint frequency in epochs (default: %(default)s)"}}
    sample_freq = {GROUP_NAME: "Other",
                   ARGS: ["-sf", "--sample-freq"],
                   KWARGS: {TYPE: int,
                            REQUIRED: True,
                            HELP: "Sampling frequency in batches (default: %(default)s)"}}
    test_freq = {GROUP_NAME: "Other",
                 ARGS: ["-tf", "--test-freq"],
                 KWARGS: {TYPE: int,
                          REQUIRED: True,
                          HELP: "Test frequency in epochs (default: %(default)s)"}}


class ResumeConfig(ConfigBuilder):
    path = {ARGS: ["path"],
            KWARGS: {TYPE: str,
                     HELP: "Path to run directory"}}
    checkpoint_epoch = {ARGS: ["--ckpt-epoch"],
                        KWARGS: {TYPE: int,
                                 DEFAULT: None,
                                 HELP: "Checkpoint epoch to load, None for latest checkpoint (default: %(default)s)"}}


class ConverterConfig(ResumeConfig):
    dyn_range_q = {ARGS: ["--dyn-range-q"],
                   KWARGS: {ACTION: "store_true",
                            HELP: "Post-training dynamic range quantization, https://www.tensorflow.org/lite/performance/post_training_quantization"}}
    int_float_q = {ARGS: ["--int-float-q"],
                   KWARGS: {ACTION: "store_true",
                            HELP: "Post-training full integer quantization, integer with float fallback"}}
    int_q = {ARGS: ["--int-q"],
             KWARGS: {TYPE: str,
                      DEFAULT: None,
                      CHOICES: ["int8", "uint8"],
                      HELP: "Post-training full integer quantization, integer only (default: %(default)s)"}}
    f16_q = {ARGS: ["--f16-q"],
             KWARGS: {ACTION: "store_true",
                      HELP: "Post-training float16 quantization"}}


if __name__ == '__main__':
    import tensorflow as tf

    GPU_DEVICES = [dev.name for dev in tf.config.list_logical_devices("GPU") + tf.config.list_logical_devices("XLA_GPU")]
    DefaultConfig.devices[KWARGS][HELP] = DefaultConfig.devices[KWARGS][HELP].format(GPU_DEVICES)
    cfg = DefaultConfig.cli()
