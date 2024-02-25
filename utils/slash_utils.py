REPLACE_SLASH = True

def replace_slash(name):

    if REPLACE_SLASH and name is not None:
        name = name.replace('/', '::')

    return name