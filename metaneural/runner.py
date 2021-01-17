from typing import *
from datetime import datetime

import tensorflow as tf
import tensorflow.keras as keras
import tflite_runtime.interpreter as tflite
from pathlib import Path

from metaneural.config import DefaultConfig, ResumeConfig, ConverterConfig


class DefaultRunner:
    def __init__(self, config: DefaultConfig, run_directory: Path):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.config = config

        ################################################################
        # Paths
        self.run_path = run_directory
        self.run_path.mkdir(exist_ok=True, parents=True)
        self.samples_path = self.run_path.joinpath("samples")
        self.model_path = self.run_path.joinpath("model")
        self.checkpoint_path = self.run_path.joinpath("checkpoint")

        ################################################################
        self._strategy = self._init_strategy()
        self.model, self.optimizer = self.init_model()
        self.quantize_model()
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)

        ################################################################
        self.dataset = self.init_dataset()
        self.repr_dataset = self.init_repr_dataset()

    ################################################################
    # https://www.tensorflow.org/api_docs/python/tf/distribute
    def _init_strategy(self) -> tf.distribute.Strategy:
        if self.config.use_tpu:
            kwargs = {} if self.config.tpu_name is None else {"tpu": self.config.tpu_name}
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(**kwargs)
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            return tf.distribute.experimental.TPUStrategy(resolver)

        else:
            all_devices = [dev.name for dev in tf.config.list_logical_devices()]
            devices = self.config.devices

            if len(devices) > 1:
                for device in devices:
                    assert device in all_devices, f"Invalid device {device}"
                return tf.distribute.MirroredStrategy(devices=devices)

            else:
                assert devices[0] in all_devices, f"Invalid device {devices[0]}"
                return tf.distribute.OneDeviceStrategy(device=devices[0])

    def with_strategy(func):
        """Run function in a strategy context"""

        def wrapper(self, *args, **kwargs):
            return self._strategy.experimental_run_v2(lambda: func(self, *args, **kwargs))

        return wrapper

    def merge(merge_fn):
        """
        Merge args across replicas and run merge_fn in a cross-replica context.
        https://www.tensorflow.org/api_docs/python/tf/distribute/ReplicaContext
        """

        def wrapper(*args, **kwargs):
            def descoper(strategy: Optional[tf.distribute.Strategy] = None, *args2, **kwargs2):
                if strategy is None:
                    return merge_fn(*args2, **kwargs2)
                else:
                    with strategy.scope():
                        return merge_fn(*args2, **kwargs2)

            return tf.distribute.get_replica_context().merge_call(descoper, args, kwargs)

        return wrapper

    ################################################################
    @with_strategy
    def init_model(self) -> Tuple[
        Union[keras.Model, Tuple[keras.Model, ...]],
        Union[keras.optimizers.Optimizer, Tuple[keras.optimizers.Optimizer, ...]]]:
        raise NotImplementedError

    def init_dataset(self) -> tf.data.Dataset:
        raise NotImplementedError

    def init_repr_dataset(self) -> Optional[Generator]:
        return None

    def quantize_model(self):
        if bool(self.config.q_aware_train[0]):
            import tensorflow_model_optimization as tfmot
            self.model = tfmot.quantization.keras.quantize_model(self.model)

    ################################################################
    @classmethod
    def new_run(cls, config: DefaultConfig):
        self = cls(config=config,
                   run_directory=Path(f"runs/{config.name}_{datetime.now().replace(microsecond=0).isoformat()}"))
        self.summary(plot=True)

        try:
            self.train()
        except KeyboardInterrupt:
            print("\nStopping...")

    @classmethod
    def resume(cls, config: DefaultConfig, run_directory: Path, checkpoint_path: Path):
        self = cls(config=config,
                   run_directory=run_directory)
        self.restore(checkpoint_path=checkpoint_path)
        self.summary()

        try:
            self.train(resume=True)
        except KeyboardInterrupt:
            print("\nStopping...")

    @classmethod
    def convert(cls, config: DefaultConfig, converter_config: ConverterConfig,
                run_directory: Path, checkpoint_path: Path):
        self = cls(config=config,
                   run_directory=run_directory)
        self.restore(checkpoint_path=checkpoint_path)

        print("Saving model")
        self.save_model()

        print("Converting model")
        self.convert_model(converter_config)

    ################################################################
    def get_callbacks(self) -> Tuple[keras.callbacks.Callback, ...]:
        tensorboard_cb = tf.keras.callbacks.TensorBoard(
            log_dir=self.run_path,
            histogram_freq=self.config.test_freq,
            update_freq="batch")
        epoch_update_cb = keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: setattr(self.config, "epoch", epoch + 1))
        checkpoint_cb = tf.keras.callbacks.LambdaCallback(
            on_epoch_begin=lambda epoch, logs: self.snap() if epoch % self.config.checkpoint_freq == 0 else None)

        return tensorboard_cb, epoch_update_cb, checkpoint_cb

    def train(self, resume: bool = False):
        raise NotImplementedError

    ################################################################
    # Saving, snapping, etc
    def summary(self, plot: bool = False):
        self._summary(self.model, "model", plot)

    def _summary(self, model: tf.keras.Model, name: str, plot: bool,
                 show_shapes: bool = False,
                 show_layer_names: bool = True,
                 rankdir: str = "TB",
                 expand_nested: bool = False,
                 dpi: int = 96):
        model.summary()
        if plot:
            tf.keras.utils.plot_model(model,
                                      to_file=self.run_path.joinpath(f"{name}.png"),
                                      show_shapes=show_shapes,
                                      show_layer_names=show_layer_names,
                                      rankdir=rankdir,
                                      expand_nested=expand_nested,
                                      dpi=dpi)

    def snap(self):
        prefix = self.checkpoint_path.joinpath(str(self.config.epoch))
        prefix.mkdir(parents=True, exist_ok=True)

        self.config.save(prefix.joinpath("config.yaml"))
        self.checkpoint.write(prefix.joinpath("model"))

    @with_strategy
    def restore(self, checkpoint_path: Path):
        self.checkpoint.restore(str(checkpoint_path / "model"))

    def save_model(self):
        self._save_model(self.model, "model")

    def _save_model(self, model: tf.keras.Model, name: str):
        model.save(str(self.model_path.joinpath(name)), save_format="tf")

    def convert_model(self, config: ConverterConfig):
        self._convert_model(self.model, "model", self.repr_dataset,
                            dyn_range_q=config.dyn_range_q, f16_q=config.f16_q,
                            int_float_q=config.int_float_q, int_q=config.int_q)

    def _convert_model(self,
                       model: tf.keras.Model,
                       name: str,
                       repr_dataset: Optional[Generator],
                       dyn_range_q: bool,
                       f16_q: bool,
                       int_float_q: bool,
                       int_q: Optional[str]):
        # TFLite
        print("TFLite conversion")
        converter = tflite.TFLiteConverter.from_keras_model(model)
        with self.model_path.joinpath(f"{name}.tflite").open("wb") as file:
            file.write(converter.convert())

        # TFLite quantizised
        # Dynamic range quantization
        if dyn_range_q:
            print("Dynamic range quantization")
            converter = tflite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tflite.Optimize.DEFAULT]
            with self.model_path.joinpath(f"{name}.dyn-range.qtflite").open("wb") as file:
                file.write(converter.convert())

        # Float16 quantization
        if f16_q:
            print("Float16 quantization")
            converter = tflite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tflite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            with self.model_path.joinpath(f"{name}.f16.qtflite").open("wb") as file:
                file.write(converter.convert())

        if repr_dataset is None:
            print(f"No representative dataset for {name}")
        else:
            # Full integer quantization, integer with float fallback
            if int_float_q:
                print("Full integer quantization, integer with float fallback")
                converter = tflite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tflite.Optimize.DEFAULT]
                converter.repr_dataset = repr_dataset
                with self.model_path.joinpath(f"{name}.int-float.qtflite").open("wb") as file:
                    file.write(converter.convert())

            # Full integer quantization, integer only
            if int_q is not None:
                print("Full integer quantization, integer only")
                converter = tflite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tflite.Optimize.DEFAULT]
                converter.repr_dataset = repr_dataset
                converter.target_spec.supported_ops = [tflite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = getattr(tf, int_q)
                converter.inference_output_type = getattr(tf, int_q)
                with self.model_path.joinpath(f"{name}.int.qtflite").open("wb") as file:
                    file.write(converter.convert())