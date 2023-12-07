from importlib import import_module


def controller_from_string(controller_str):
    return ControllerFactory(controller_str=controller_str)


class ControllerFactory:
    # noinspection SpellCheckingInspection
    valid_base_controllers = {
        "mpc-icem": (".mpc", "MpcICem"),
        "mpc-icem-torch": (".mpc_torch", "TorchMpcICem"),
        "mpc-curiosity-icem-torch": (".mpc_torch_curiosity", "TorchCuriosityMpcICem"),
        "mpc-rnd-icem-torch": (".mpc_torch_rnd", "TorchRNDMpcICem"),
        "mpc-rnd-ensemble-icem-torch": (".mpc_torch_rnd_ensemble", "TorchRNDMpcICem"),
        "mpc-compression-icem-torch": (
            ".mpc_torch_compression",
            "TorchCompressionMpcICem",
        ),
        "mpc-relational-rair-icem-torch": (
            ".mpc_torch_relational_rair",
            "TorchRelationalRairMpcICem",
        ),
        "mpc-distance-rair-icem-torch": (
            ".mpc_torch_distance_rair",
            "TorchRelationalRairMpcICem",
        ),
        "mpc-direct-rair-icem-torch": (
            ".mpc_torch_direct_rair",
            "TorchDirectRairMpcICem",
        ),
    }

    controller = None

    def __new__(cls, *, controller_str):

        if controller_str in cls.valid_base_controllers:
            ctrl_package, ctrl_class = cls.valid_base_controllers[controller_str]
            module = import_module(ctrl_package, "mbrl.controllers")
            cls.controller = getattr(module, ctrl_class)
        else:
            raise ImportError(
                f"cannot find '{controller_str}' in known controller: "
                f"{cls.valid_base_controllers.keys()}"
            )

        return cls.controller
