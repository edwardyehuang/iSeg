
backbone_registry_dict = {}


def register_backbone(backbone_class, name=None):

    if name is None:
        name = backbone_class.__name__
    
    if name not in backbone_registry_dict:
        backbone_registry_dict[name] = backbone_class