def values_to_list (values):

    if isinstance(values, tuple):
        values = list(values)

    if not isinstance(values, list):
        values = [values]

    return values


def values_to_tuple (values):

    values = values_to_list(values)

    if len(values) == 1:
        return values[0]

    return tuple(values)