import tensorflow as tf

def safed_softmax (x):
    t = x.dtype

    if t == tf.float16:
        x = tf.cast(x, tf.float32)
        x = tf.nn.softmax(x)
        x = tf.cast(x, t)
    else:
        x = tf.nn.softmax(x)

    return x