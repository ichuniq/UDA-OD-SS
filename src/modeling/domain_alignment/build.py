"""
To support different types of DA heads
"""
from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry

DAHEAD_REGISTRY = Registry("DAHEAD")
DAHEAD_REGISTRY.__doc__ = """
return domain alignment head 
"""


def build_da_head(cfg, input_shape=None):

    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    name = cfg.MODEL.DA.NAME
    da_head = DAHEAD_REGISTRY.get(name)(cfg, input_shape)
    return da_head