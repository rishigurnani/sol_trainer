from os.path import exists
from sol_trainer.os_utils import path_join, makedirs
from pickle import dump

from . import constants


def save_classlabels(class_labels, root_dir):
    """
    Save a dictionary of class_labels, where each key is a
    class name (e.g., "class0") and each value is a class label
    (e.g., 0).
    """
    class_labels = sorted(class_labels.items(), key=lambda x: x[0])
    class_labels = [f"{x[0]} {x[1]}" for x in class_labels]
    text = "\n".join(class_labels)
    path = path_join(root_dir, constants.METADATA_DIR, constants.CLASSLABELS_FILENAME)
    safe_save(text, path, "text")


def safe_save(object, path, save_method):
    """
    Safely save objects to a path if the path does not already exist
    Keyword arguments:
        object
        path (str)
        save_method (str): 'pickle' or 'text'
    """
    if not exists(path):
        if save_method == "pickle":
            with open(path, "wb") as f:
                dump(
                    obj=object,
                    file=f,
                    protocol=constants.PICKLE_PROTOCOL,
                )
        elif save_method == "text":
            with open(path, "w") as f:
                f.write(object)
    else:
        raise ValueError(f"{path} already exists. Object was not saved.")


def prepare_root(path_to_root):
    model_dir, md_dir = get_root_subdirs(path_to_root)
    makedirs(path_to_root)
    makedirs(model_dir)
    makedirs(md_dir)

    return model_dir, md_dir


def get_root_subdirs(path_to_root):
    model_dir = path_join(path_to_root, constants.MODELS_DIR)
    md_dir = path_join(path_to_root, constants.METADATA_DIR)

    return model_dir, md_dir
