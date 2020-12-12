from typing import Tuple, Any, Callable
from argparse import ArgumentParser
import json

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

        for field, scheme, value in self.get_field_scheme_value():
            setattr(self, field, None)
            setattr(self.__class__, field, self.set_defaults(scheme))

    def get_attrs(self) -> str:
        all_vars = {**vars(self.__class__)}
        for cls in self.__class__.__bases__:
            if cls.__name__ == ConfigBuilder.__name__:
                break
            else:
                all_vars = {**all_vars, **vars(cls)}

        for attr, value in all_vars.items():
            if not (attr.startswith("__") and attr.endswith("__")) and not isinstance(value, Callable):
                yield attr

    @staticmethod
    def set_defaults(scheme: dict) -> dict:
        """Set all default values for a scheme"""
        scheme.setdefault(SAVE, True)
        return scheme

    def get_field_scheme_value(self) -> Tuple[str, dict, Any]:
        """Fetches all available fields, schemes and values"""
        for attr in self.get_attrs():
            yield attr, getattr(self.__class__, attr), getattr(self, attr)

    @classmethod
    def cli(cls, description: str = ""):
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
                groups.setdefault(scheme[GROUP_NAME], parser.add_argument_group(scheme[GROUP_NAME]))
                target_parser = groups[scheme[GROUP_NAME]]

            else:
                target_parser = parser

            if scheme[ARGS][0].startswith("-"):
                scheme[KWARGS]["dest"] = field
            target_parser.add_argument(*scheme[ARGS], **scheme[KWARGS])

        ################################################################
        # Parse
        args = parser.parse_args()
        for field, value in vars(args).items():
            setattr(self, field, value)

        return self

    @classmethod
    def load(cls, path: Path):
        self = cls()
        self._cleanup()

        with path.open("r") as file:
            data = json.load(file)

        for field, scheme, value in self.get_field_scheme_value():
            try:
                setattr(self, field, data[field])
            except KeyError:
                raise KeyError(f"Config is missing required key '{field}'")

        return self

    def save(self, path: Path):
        self._cleanup()
        with path.open("w") as file:
            json.dump(self.to_json(), file, indent=4)

    def to_json(self) -> dict:
        data = {}
        for field, scheme, value in self.get_field_scheme_value():
            data[field] = getattr(self, field)
        return data

    def __repr__(self):
        return self.to_json()

    def _cleanup(self):
        to_remove = []
        for field, scheme, value in self.get_field_scheme_value():
            if not scheme[SAVE]:
                to_remove.append(field)

        for field in to_remove:
            delattr(self, field)
            delattr(self.__class__, field)


class DefaultConfig(ConfigBuilder):
    """
    This config implementation allows to easily trace param usage with the help of IDE

    Examples:
    name = {GROUP_NAME: "Model",                                       # Not required
            ARGS: ["--name"],                                          # Required
            KWARGS: {TYPE: str, REQUIRED: True, HELP: "Model name"}},  # Required
            SAVE: False                                                # Saving in config.json, gets wiped on save (default: True)

    # Does not provide cli param, just exists
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

    # Device params
    use_tpu = {GROUP_NAME: "Device params",
               ARGS: ["--use-tpu"],
               KWARGS: {ACTION: "store_true",
                        HELP: "Use Google Cloud TPU, if True, --gpu param is ignored (default: %(default)s)"}}
    tpu_name = {GROUP_NAME: "Device params",
                ARGS: ["--tpu-name"],
                KWARGS: {TYPE: str, DEFAULT: None,
                         HELP: "Google Cloud TPU name, if None and flag --use-tpu is set, will try to detect automatically (default: %(default)s)"}}
    devices = {GROUP_NAME: "Device params",
               ARGS: ["--gpu"],
               KWARGS: {TYPE: str, DEFAULT: None,
                        HELP: "Available GPUs: {}, list devices with , as delimiter"}}  # Format however is needed
    xla_jit = {GROUP_NAME: "Device params",
               ARGS: ["--xla-jit"],
               KWARGS: {ACTION: "store_true",
                        HELP: "XLA Just In Time compilation, https://www.tensorflow.org/xla (default: %(default)s)"}}

    # Training params
    epoch = {CONSTANT: 0}
    epochs = {GROUP_NAME: "Training params",
              ARGS: ["-e", "--epochs"],
              KWARGS: {TYPE: int,
                      REQUIRED: True,
                      HELP: "Epochs (default: %(default)s)"}}
    batch_size = {GROUP_NAME: "Dataloader params",
                  ARGS: ["-b", "--batch-size"],
                  KWARGS: {TYPE: int, DEFAULT: 1, HELP: "Batch size (default: %(default)s)"}}
    q_aware_train = {GROUP_NAME: "Training params",
                     ARGS: ["-qat", "--quantization-aware-training"],
                     KWARGS: {NARGS: "+",
                              TYPE: int,
                              DEFAULT: [0],
                              HELP: "Quantization aware training for chosen models, https://www.tensorflow.org/model_optimization/guide/quantization/training (default: %(default)s)"}}
    test_split = {GROUP_NAME: "Training params",
                  ARGS: ["-ts"],
                  KWARGS: {TYPE: float,
                           DEFAULT: 0.1,
                           HELP: "Test split (default: %(default)s)"}}

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


class ConverterConfig(ConfigBuilder):
    path = {ARGS: ["path"],
            KWARGS: {TYPE: str,
                     HELP: "Path to run directory"}}
    checkpoint_epoch = {ARGS: ["--ckpt-epoch"],
                        KWARGS: {TYPE: int,
                                 DEFAULT: None,
                                 HELP: "Checkpoint epoch to load, None for latest checkpoint (default: %(default)s)"}}

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
    cfg = DefaultConfig.cli()
