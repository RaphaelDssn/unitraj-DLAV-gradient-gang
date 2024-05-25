from motionnet.models.ptr.ptr import PTR
from motionnet.models.HPNet.hpnet import HPNet

__all__ = {
    'ptr': PTR,
    'hpnet': HPNet,
}


def build_model(config):

    model = __all__[config.method.model_name](
        config=config
    )

    return model
