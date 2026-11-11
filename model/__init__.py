from model.model import Model_base
from model.model_cl.model_cl import Model_cl


_EXPORTS = {
    "Pretrain",
    "Retrain",
    "NewNode",
    "AllNode",
    "EAC",
    "trafficstream",
    "STBP",
    "STPrompt",
}


def __getattr__(name):
    if name in {"Pretrain", "Retrain", "NewNode", "AllNode"}:
        from model.model_cl.naive import AllNode, NewNode, Pretrain, Retrain

        exports = {
            "Pretrain": Pretrain,
            "Retrain": Retrain,
            "NewNode": NewNode,
            "AllNode": AllNode,
        }
    elif name == "EAC":
        from model.model_cl.eac import Model as EAC

        exports = {"EAC": EAC}
    elif name == "trafficstream":
        from model.model_cl.trafficstream import Model as trafficstream

        exports = {"trafficstream": trafficstream}
    elif name == "STBP":
        from model.model_cl.stbp import Model as STBP

        exports = {"STBP": STBP}
    elif name == "STPrompt":
        from model.model_cl.stprompt import Model as STPrompt

        exports = {"STPrompt": STPrompt}
    else:
        raise AttributeError(f"module 'model' has no attribute {name!r}")

    globals().update(exports)
    return exports[name]


__all__ = ["Model_base", "Model_cl", *_EXPORTS]
