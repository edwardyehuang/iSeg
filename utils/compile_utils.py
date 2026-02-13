import keras


def autograph_do_not_convert(func):
    """Decorator that suppresses the conversion of a function.

    Args:
        func: function to decorate.

    Returns:
        If `func` is not None, returns a `Callable` which is equivalent to
        `func`, but is not converted by AutoGraph.
        If `func` is None, returns a decorator that, when invoked with a
        single `func` argument, returns a `Callable` equivalent to the
        above case.
    """

    backend = keras.backend.backend()

    if backend == "tensorflow":
        import tensorflow as tf # pylint: disable=import-outside-toplevel
        return tf.autograph.experimental.do_not_convert(func)