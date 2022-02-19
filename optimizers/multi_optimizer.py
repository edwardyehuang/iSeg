# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

# Motified from https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/discriminative_layer_training.py

import tensorflow as tf

class MultiOptimizer(tf.keras.optimizers.Optimizer):
    
    def __init__(
        self,
        optimizers_and_layers=None,
        optimizer_specs=None,
        name="MultiOptimizer",
        **kwargs,
    ):

        super(MultiOptimizer, self).__init__(name, **kwargs)

        if optimizer_specs is None and optimizers_and_layers is not None:
            self.optimizer_specs = [
                self.create_optimizer_spec(optimizer, layers_or_model)
                for optimizer, layers_or_model in optimizers_and_layers
            ]

        elif optimizer_specs is not None and optimizers_and_layers is None:
            self.optimizer_specs = [
                self.maybe_initialize_optimizer_spec(spec) for spec in optimizer_specs
            ]

        else:
            raise RuntimeError(
                "Must specify one of `optimizers_and_layers` or `optimizer_specs`."
            )

    def apply_gradients(self, grads_and_vars, name, **kwargs):
        """Wrapped apply_gradient method.
        Returns an operation to be executed.
        """

        for spec in self.optimizer_specs:
            spec["gv"] = []

        for grad, var in tuple(grads_and_vars):
            for spec in self.optimizer_specs:
                for weight_name in spec["weights"]:
                    if var.name == weight_name:
                        spec["gv"].append((grad, var))

        update_ops = []

        for spec in self.optimizer_specs:
            update_ops += [spec["optimizer"].apply_gradients(spec["gv"], name, **kwargs)]

        if len(update_ops) == 1:
            return update_ops[0]

        with tf.control_dependencies(update_ops[:-1]):
            return update_ops[-1]


    def get_config(self):
        config = super(MultiOptimizer, self).get_config()
        config.update({"optimizer_specs": self.optimizer_specs})
        return config

    @classmethod
    def create_optimizer_spec(
        cls,
        optimizer,
        layers_or_model
    ):
        """Creates a serializable optimizer spec.
        The name of each variable is used rather than `var.ref()` to enable serialization and deserialization.
        """
        if isinstance(layers_or_model, list):
            weights = [
                var.name for sublayer in layers_or_model for var in sublayer.weights
            ]
        else:
            weights = [var.name for var in layers_or_model.weights]

        return {
            "optimizer": optimizer,
            "weights": weights,
        }

    @classmethod
    def maybe_initialize_optimizer_spec(cls, optimizer_spec):
        if isinstance(optimizer_spec["optimizer"], dict):
            optimizer_spec["optimizer"] = tf.keras.optimizers.deserialize(
                optimizer_spec["optimizer"]
            )

        return optimizer_spec

    def __repr__(self):
        return "Multi Optimizer with %i optimizer layer pairs" % len(
            self.optimizer_specs
        )