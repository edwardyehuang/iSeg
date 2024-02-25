REPLACE_SLASH = False

def replace_slash(name):

    if REPLACE_SLASH:
        name = name.replace('/', '::')

    return name