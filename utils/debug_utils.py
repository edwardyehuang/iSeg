import keras


def assert_scalar(tensor, message=None):
    """Asserts that the given `tensor` is a scalar.

    This function raises `ValueError` unless it can be certain that the given
    `tensor` is a scalar. `ValueError` is also raised if the shape of `tensor` is
    unknown.

    This is always checked statically, so this method returns nothing.

    Args:
        tensor: A `Tensor`.
        message: A string to prefix to the default message.

    Raises:
        ValueError: If the tensor is not scalar (rank 0), or if its shape is
        unknown.
    """

    backend = keras.backend.backend()

    if backend == "tensorflow":
        import tensorflow as tf # pylint: disable=import-outside-toplevel
        
        return tf.debugging.assert_scalar(tensor, message=message)
    
    return tensor