"""Bridge the upstream transform hooks to the installed torchvision v2 API.

Older torchvision calls _transform/_get_params, newer releases call
transform/make_params. Restrict aliases to this repository's custom classes.
"""
from torchvision.transforms.v2 import Transform
from engine.data.transforms import _transforms

for cls in vars(_transforms).values():
    if isinstance(cls,type) and issubclass(cls,Transform) and cls.__module__==_transforms.__name__:
        if '_transform' in cls.__dict__ and 'transform' not in cls.__dict__:
            cls.transform=cls.__dict__['_transform']
        if '_get_params' in cls.__dict__ and 'make_params' not in cls.__dict__:
            cls.make_params=cls.__dict__['_get_params']

