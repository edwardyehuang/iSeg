import tensorflow as tf

def values_to_list (values):

    if isinstance(values, tuple):
        values = list(values)

    if not isinstance(values, list):
        values = [values]

    return values