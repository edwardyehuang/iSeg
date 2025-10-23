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



def add_list_to_flattened_dict (key_prefix, values, dict_obj=None):

    if dict_obj is None:
        dict_obj = {}

    values = values_to_list(values)

    for i, value in enumerate(values):
        dict_obj[f"{key_prefix}_{i}"] = value

    return dict_obj


def extract_list_from_flattened_dict (key_prefix, dict_obj):

    values = []

    i = 0
    while True:
        key = f"{key_prefix}_{i}"
        if key not in dict_obj:
            break

        values.append(dict_obj[key])
        i += 1

    return values