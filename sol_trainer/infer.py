import numpy as np
from torch import device as torch_device
from torch import cuda
from torch_geometric.loader import DataLoader
from dataclasses import dataclass
import warnings

from sol_trainer import constants
from sol_trainer.hyperparameters.hyperparameters import compute_batch_max
from sol_trainer.load import load_hps, load_selectors
from sol_trainer.utils import _assemble_data
from .prepare import prepare_infer


def modulate_dropout(model, mode):
    """
    Function to enable the dropout layers during test-time.
    Taken from https://stackoverflow.com/questions/63285197/measuring-uncertainty-using-mc-dropout-on-pytorch
    """
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            if mode == "train":
                m.train()
            elif mode == "test":
                m.eval()
            else:
                raise ValueError("Invalid option passed in for mode")


def _model_eval_mode(model, dropout_mode):
    """
    Function to control how the model behaves in
    eval mode
    """
    if isinstance(model, dict):
        for _, v in model.items():
            __model_eval_mode(v, dropout_mode)
    elif isinstance(model, list):
        for v in model:
            __model_eval_mode(v, dropout_mode)
    elif hasattr(model, "forward"):
        __model_eval_mode(model, dropout_mode)
    else:
        raise TypeError(f"Invalid type {type(model)} used for keyword argument model.")


def __model_eval_mode(model, dropout_mode):
    model.eval()
    if dropout_mode == "train":
        modulate_dropout(model, "train")


def init_evaluation(model):
    y_val = []  # true labels
    y_val_hat_mean = []  # prediction mean
    y_val_hat_std = []  # prediction uncertainty
    selectors = []
    model.eval()

    return y_val, y_val_hat_mean, y_val_hat_std, selectors


def eval_ensemble(
    model,
    root_dir,
    dataframe,
    smiles_featurizer,
    device=torch_device("cuda" if cuda.is_available() else "cpu"),
    ensemble_forward_kwargs={},
    ensemble_kwargs_dict={},
    eval_config=None,
    ncore=1,
    return_dict=False,
):
    """
    Evaluate ensemble on the data contained in dataframe.

    Keyword arguments:
        model (nn.Module): The ensemble to be evaluated
        root_dir (str): The path to the directory containing the ensemble
            information
        dataframe (pd.DataFrame): The data to evaluate, in melted form.
        smiles_featurizer
        device (torch.device)
        ensemble_forward_kwargs (dict): Arguments to pass into the
            forward method of the ensemble.
        ensemble_kwargs_dict: (Deprecated since June 30th, 2022) Arguments
            to pass into the forward method of the ensemble.
        eval_config (evalConfig):
        ncore (int): Number of cores to use during smiles_featurization.
        return_dict (bool): If True, data will be returned in a
            dictionary.

    Outputs:
        y (np.ndarray): Data labels
        y_hat_mean (np.ndarray): Mean of data predictions
        y_hat_std (np.ndarray): Std. dev. of data predictions
        y_selectors (np.ndarray): Selector for each data point
    """
    # Handle outdated code that uses ensemble_kwargs_dict.
    if ensemble_kwargs_dict:
        warnings.warn(
            "The argument 'ensemble_kwargs_dict' has been deprecated in favor of 'ensemble_forward_kwargs'.",
            DeprecationWarning,
            # stacklevel=2,
        )
        ensemble_forward_kwargs.update(ensemble_kwargs_dict)

    # If eval_config is None, we will generate it.
    if not eval_config:
        # If we are using a CPU, do not use max batch.
        if str(device) == "cpu":
            eval_config = evalConfig(use_max_batch=False)
        # If we are not using a CPU, we are using CUDA. So use max batch.
        else:
            eval_config = evalConfig(use_max_batch=True)
    model.to(device)
    selectors = load_selectors(root_dir)
    selector_dim = len(selectors)
    # Prepare dataframe.
    if "data" not in dataframe.columns:
        hps = load_hps(root_dir)
        dataframe = prepare_infer(
            dataframe,
            smiles_featurizer,
            root_dir=root_dir,
            scale_labels=False,
            hps=hps,
            ncore=ncore,
        )
    else:
        raise ValueError("dataframe should not contain any key(s) named 'data'.")
    data_ls = dataframe.data.values.tolist()
    # If eval_config has an attribute named "batch_size", we will just use
    # that. Otherwise, we will need to compute it.
    if not hasattr(eval_config, "batch_size"):
        if eval_config.use_max_batch:
            eval_config.batch_size = compute_batch_max(
                model,
                data_ls,
                device,
                forward_kwargs=ensemble_forward_kwargs,
                init_size=len(data_ls),
            )
            print(
                "The supplied evalConfig did not have an assigned "
                + "batch size, as a result, the max batch size "
                + f"of {eval_config.batch_size} will be used."
            )
        else:
            # TODO: Deprecate this?
            eval_config.batch_size = constants.BS_MAX
    loader = DataLoader(data_ls, batch_size=eval_config.batch_size, shuffle=False)

    return _evaluate_ensemble(
        model, loader, device, selector_dim, return_dict, **ensemble_forward_kwargs
    )


def _evaluate_ensemble(model, val_loader, device, selector_dim, return_dict, **kwargs):
    """
    Evaluate ensemble on the data contained in val_loader. This function is not
    to be called directly. It is a helper function for eval_ensemble.
    Keyword arguments:
        model (nn.Module): The ensemble to be evaluated.
        val_loader (DataLoader): The data to evaluate.
        device (torch.device)
        selector_dim (int): The number of selector dimensions.
        return_dict (bool): If True, data will be returned in a
            dictionary.
        **kwargs: Arguments to pass into the 'forward' method of model.
    Outputs:
        y (np.ndarray): Data labels
        y_hat_mean (np.ndarray): Mean of data predictions
        y_hat_std (np.ndarray): Std. dev. of data predictions
        y_selectors (np.ndarray): Selector for each data point
    """
    return _evaluate(
        model, val_loader, device, True, selector_dim, return_dict, **kwargs
    )


def eval_submodel(model, val_loader, device, selector_dim=None):
    """
    Evaluate model on the data contained in val_loader.

    Outputs:
        y (np.ndarray): Data labels
        y_hat (np.ndarray): Data predictions
        y_selectors (np.ndarray): Selector for each data point
    """
    return _evaluate(
        model, val_loader, device, monte_carlo=False, selector_dim=selector_dim
    )


def _evaluate(
    model, val_loader, device, monte_carlo, selector_dim, return_dict=False, **kwargs
):
    """
    Evaluate model on the data contained in val_loader. This function is not
    to be called directly. It is a helper function for eval_submodel and
    eval_ensemble.

    Keyword arguments:
        model (nn.Module): The ensemble to be evaluated.
        val_loader (DataLoader): The data to evaluate.
        device (torch.device)
        selector_dim (int): The number of selector dimensions.
        return_dict (bool): If True, data will be returned in a
            dictionary.
        **kwargs: Arguments to pass into the 'forward' method of model.
    """

    y_val, y_val_hat_mean, y_val_hat_std, selectors = init_evaluation(model)
    for ind, data in enumerate(val_loader):  # loop through validation batches
        data = data.to(device)
        # sometimes the batch may have labels associated. Let's check
        if data.y is not None:
            y_val += data.y.detach().cpu().numpy().tolist()
        # sometimes the batch may have selectors associated. Let's check
        if selector_dim:
            selectors += data.selector.cpu().numpy().tolist()
        if not monte_carlo:
            data = model(_assemble_data(model, data))
            y_val_hat_mean += data.yhat.detach().cpu().numpy().tolist()
        # if we are doing a monte_carlo evaluation then we will have two
        # outputs: the mean and standard deviation.
        else:
            # If we want to use MC dropout then we need to keep dropout in
            # train mode. To do this, we need to pass both the Ensemble and
            # each of its submodels through "_model_eval_mode".
            if model.monte_carlo:
                dropout_mode = "train"
            else:
                dropout_mode = "test"
            _model_eval_mode(model, dropout_mode=dropout_mode)
            for submodel in model.submodel_dict.values():
                _model_eval_mode(submodel, dropout_mode=dropout_mode)
            data = model(data, **kwargs)
            y_val_hat_mean += data.yhat.flatten().detach().cpu().numpy().tolist()
            y_val_hat_std += data.yhat_std.flatten().detach().cpu().numpy().tolist()
    del data  # free memory
    y_val, y_val_hat_mean = np.array(y_val), np.array(y_val_hat_mean)
    d = {
        "y_val": y_val,
        "y_val_hat_mean": y_val_hat_mean,
        "selectors": selectors,
        "y_val_hat_std": y_val_hat_std,
    }
    if monte_carlo:
        y_val_hat_std = np.array(y_val_hat_std)
        if return_dict:
            d.update(
                {
                    "y_val_hat_std": y_val_hat_std,
                }
            )
            return d
        else:
            return (
                y_val,
                y_val_hat_mean,
                y_val_hat_std,
                selectors,
            )
    else:
        if return_dict:
            return d
        else:
            return y_val, y_val_hat_mean, selectors


@dataclass
class evalConfig:
    """
    A class to pass into eval* and _eval* functions.
    """

    # need to be set manually
    use_max_batch: bool
