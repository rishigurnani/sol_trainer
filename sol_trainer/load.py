from sol_trainer.hyperparameters.hyperparameters import HpConfig
from sol_trainer.os_utils import path_join
from os import listdir, path
import pickle
from torch import load as torch_load
from re import search, compile

from . import constants as ks
from . import models


def file_filter(root_dir, pattern):
    if isinstance(pattern, str):
        pattern = compile(pattern)
    return [path_join(root_dir, f) for f in listdir(root_dir) if search(pattern, f)]


def load_model(path, submodel_cls, **kwargs):
    model = submodel_cls(
        **kwargs,
    )
    model.load_state_dict(torch_load(path))
    return model


def safe_pickle_load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_selectors(root_dir):
    selectors_path = path_join(root_dir, ks.METADATA_DIR, ks.SELECTORS_FILENAME)
    return safe_pickle_load(selectors_path)


def load_features(root_dir):
    path = path_join(root_dir, ks.METADATA_DIR, ks.FEATURE_FILENAME_PKL)
    return safe_pickle_load(path)


def load_scalers(root_dir):
    scalers_path = path_join(root_dir, ks.METADATA_DIR, ks.SCALERS_FILENAME)
    return safe_pickle_load(scalers_path)


def load_hps(root_dir):
    hps_path = path_join(root_dir, ks.METADATA_DIR, ks.HPS_FILENAME)
    loaded_hps = safe_pickle_load(hps_path)
    hps = HpConfig()
    if loaded_hps:
        # If hyperparameters were saved, let's load them into hps.
        hps.set_values_from_string(str(loaded_hps))

    return hps


def load_submodel_dict(root_dir, submodel_cls, submodel_kwargs_dict):
    model_dir = path_join(root_dir, ks.MODELS_DIR)
    submodel_paths = sorted(file_filter(model_dir, ks.submodel_re))
    # load hps
    if "hps" not in submodel_kwargs_dict:
        hps_path = path_join(root_dir, ks.METADATA_DIR, ks.HPS_FILENAME)
        with open(hps_path, "rb") as f:
            submodel_kwargs_dict["hps"] = pickle.load(f)
    submodel_dict = {
        ind: load_model(path, submodel_cls, **submodel_kwargs_dict)
        for ind, path in enumerate(submodel_paths)
    }

    return submodel_dict


def load_ensemble(
    ensemble_class,
    root_dir,
    submodel_cls,
    device,
    submodel_kwargs_dict,
    ensemble_init_kwargs={},
):
    """
    Load the ensemble from root_dir. The ensemble type will
    be inferred from the contents of root_dir

    Keywords args
        root_dir: The path to the directory containing the model information.
        submodel_cls: The class corresponding to the submodel to load
        device (torch.device):
        submodel_kwargs_dict: Other arguments needed to instantiate the
            submodel.
        ensemble_init_kwargs (dict): Arguments to pass into __init__
            method of the ensemble.
    """
    # Load the scalers.
    scalers_path = path_join(root_dir, ks.METADATA_DIR, ks.SCALERS_FILENAME)
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)
    # Load the submodels.
    submodel_dict = load_submodel_dict(root_dir, submodel_cls, submodel_kwargs_dict)
    # Send each submodel to the appropriate device.
    for model in submodel_dict.values():
        model.to(device)
    # Check if "regression" should be True or False.
    classlabels_path = path_join(root_dir, ks.METADATA_DIR, ks.CLASSLABELS_FILENAME)
    if path.exists(classlabels_path):
        regression = False
    else:
        regression = True

    if ("regression" in ensemble_init_kwargs) and (
        ensemble_init_kwargs["regression"] != regression
    ):
        if regression:
            suffix = "does not exist"
        else:
            suffix = "exists"
        raise ValueError(
            f"The value passed in to ensemble_init_kwargs['regression'] is "
            + f"{ensemble_init_kwargs['regression']} but {classlabels_path} {suffix}."
        )
    else:
        ensemble_init_kwargs["regression"] = regression
    # Instantiate the ensemble.
    if ensemble_class == models.LinearEnsemble:
        if submodel_dict:
            ensemble = models.LinearEnsemble(
                submodel_dict,
                device,
                scalers,
                **ensemble_init_kwargs,
            )
        else:
            raise ValueError("No submodels found.")

    return ensemble


def load_classlabels(root_dir, reverse=False):
    """
    Load a dictionary of class_labels. If reverse is True, the
    dictionary will be constructed in the reverse order compared
    to which it was saved. That is, the keys will be class labels (e.g.,
    0) and the values will be class names (e.g., class0). If reverse is
    False, then the keys will be class names and the values will be class
    labels.
    """
    path = path_join(root_dir, ks.METADATA_DIR, ks.CLASSLABELS_FILENAME)
    with open(path, "r") as f:
        text = f.readlines()
    class_labels = {}
    for line in text:
        line = line.strip()
        line = line.split(" ")
        label = int(line[-1])
        name = " ".join(line[:-1])
        if reverse:
            class_labels[label] = name
        else:
            class_labels[name] = label
    return class_labels
