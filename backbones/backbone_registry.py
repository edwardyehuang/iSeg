
backbone_registry_dict = {}


def register_backbone(backbone_class, name=None):

    if name is None:
        name = backbone_class.__name__
        name = name.lower()

    if isinstance(name, tuple):
        name = list(name)

    if not isinstance(name, list):
        name = [name]

    for n in name:
        if n not in backbone_registry_dict:
            backbone_registry_dict[n] = backbone_class