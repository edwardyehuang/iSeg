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


def values_to_tuple_2d (values):

    values = values_to_list(values)

    if len(values) == 1:
        values *= 2

    assert len(values) == 2

    return tuple(values)