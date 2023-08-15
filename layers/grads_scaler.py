import tensorflow as tf

def scale_grads (x, scale_rate=1.0):
    @tf.custom_gradient
    def scale_grads_func (x):

        def grad (upstream):

            return upstream * tf.cast(scale_rate, upstream.dtype)
        
        return x, grad
    
    return scale_grads_func(x)