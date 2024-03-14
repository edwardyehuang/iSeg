import numpy as np
import tensorflow as tf
import ml_dtypes

from h5py._hl.dataset import Dataset

def h5py_dataset_converter(value, dtype=None, name=None, as_ref=False):
    if "dtype" in value.attrs and value.attrs["dtype"] == "bfloat16":
        value = np.array(value, dtype=ml_dtypes.bfloat16)
    else:
        value = np.array(value)
    
    value = tf.convert_to_tensor(value, dtype=dtype, name=name)
    
    return value


tf.register_tensor_conversion_function(
    Dataset, 
    h5py_dataset_converter,
)
