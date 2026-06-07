from dataloader.dataloader import Data_Base, Data_CL


_EXPORTS = {
    "Data_pems_cl",
    "Data_nyc_cl",
    "Data_air_cl",
}


def __getattr__(name):
    if name == "Data_pems_cl":
        from dataloader.pems_cl import Data_pems_cl

        obj = Data_pems_cl
    elif name == "Data_nyc_cl":
        from dataloader.nyc_cl import Data_nyc_cl

        obj = Data_nyc_cl
    elif name == "Data_air_cl":
        from dataloader.air_cl import Data_air_cl

        obj = Data_air_cl
    else:
        raise AttributeError(f"module 'dataloader' has no attribute {name!r}")

    globals()[name] = obj
    return obj


__all__ = ["Data_Base", "Data_CL", *_EXPORTS]
