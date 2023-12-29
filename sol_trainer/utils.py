import os
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score
from math import nan
import random
import numpy as np
import torch
from datetime import datetime
import time
import GPUtil
import warnings

import sol_trainer.constants as ks

# fix random seed
random.seed(2)
torch.manual_seed(2)
np.random.seed(2)


class DummyScaler:
    """
    This is a "Scaler" which just returns the object passed in.
    """

    def __init__(self):
        pass

    def transform(self, data):
        """
        Just return the data that is passed in
        """
        return data

    def inverse_transform(self, data):
        return data


def batch_scale_back(y, y_hat, scalers, selectors):
    warnings.warn(
        "utils.batch_scale_back is deprecated. Use utils.compute_per_property_metrics instead."
    )
    return_y = np.zeros(y.shape)
    return_y_hat = np.zeros(y_hat.shape)
    n_props = len(scalers)
    names = list(scalers.keys())
    for ind in range(n_props):
        name = names[ind]
        scaler = scalers[name]
        data_subset = [j for j, x in enumerate(selectors) if x[ind] != 0.0]
        y_subset = np.expand_dims(y[data_subset], 0)  # scaler needs 2D array
        y_hat_subset = np.expand_dims(y_hat[data_subset], 0)  # scaler needs 2D array
        return_y[data_subset] = scaler.inverse_transform(y_subset).squeeze()
        return_y_hat[data_subset] = scaler.inverse_transform(y_hat_subset).squeeze()
    return return_y, return_y_hat


def print_per_property_metrics(
    y_val,
    y_val_hat,
    selectors_val,
    scalers: dict,
    inverse_transform: bool,
    regression: bool = True,
):
    """
    Compute and report error metrics of model predictions, separated out
    by each property in "scalers"

    Keyword arguments:
        y_val (iterable): Labels
        y_val_hat (iterable): Model predictions
        selectors_val (iterable): A collection of selector vectors
        scalers (dict)
        inverse_transform (bool): If true, scale (y_val, y_val_hat) back
            to the original scale of each property using the
            inverse_transform method of the correct scaler in scalers.
        regression (bool): If true, only print metrics for regression tasks.
            Else print metrics for classification tasks.
    """
    property_names = list(scalers.keys())
    if regression:
        metric_fn = compute_regression_metrics
    else:
        metric_fn = compute_classification_metrics
    error_dict = compute_per_property_metrics(
        y_val,
        y_val_hat,
        selectors_val,
        property_names,
        debug=False,
        metric_fn=metric_fn,
        inverse_transform=inverse_transform,
        scalers=scalers,
    )
    for key in error_dict:
        if regression:
            print(
                f"[{key} orig. scale val rmse] {error_dict[key][0]} [{key} orig. scale val r2] {error_dict[key][1]}",
                flush=True,
            )
        else:
            print(
                f"[{key} orig. scale val acc] {error_dict[key][0]} [{key} orig. scale val f1] {error_dict[key][1]}",
                flush=True,
            )


def compute_regression_metrics(y, y_hat, mt, round_to=3):
    try:
        rmse = round(np.sqrt(mean_squared_error(y, y_hat)), round_to)
        r2 = round(r2_score(y, y_hat), round_to)
    except ValueError as e:
        print((y, y_hat))
        raise e
    return rmse, r2


def compute_classification_metrics(y, y_hat, mt, round_to=3):
    if y.shape != y_hat.shape:
        y_hat = np.argmax(y_hat, 1)
    try:
        acc = round(accuracy_score(y, y_hat), round_to)
        f1 = round(f1_score(y, y_hat, average="macro"), round_to)
    except ValueError as e:
        print((y, y_hat))
        raise e
    return acc, f1


def compute_per_property_metrics(
    y,
    y_hat,
    selectors,
    property_names,
    debug=False,
    metric_fn=compute_regression_metrics,
    inverse_transform=True,
    scalers=None,
):
    """
    Compute regression metrics for several properties separately. Return
    the metrics as a dictionary of type {name: (rmse, r2)}

    Keyword Arguments and types:
    y - a *numpy array*
    y_hat - a *numpy array*
    selectors - list of *lists*
    property_names - list of *strings*. The order of strings
        should match the selector dimension
    """
    if debug:
        print(y[0:5])
        print(y_hat[0:5])
        print(selectors[0:5])
        print(property_names)

    return_dict = {}
    n_props = len(property_names)
    for ind in range(n_props):
        name = property_names[ind]
        # If we have more than one property, we need to compute the subset
        # of (y, y_hat, selectors) that correspond to "name". If we only have
        # one property, then we know that every point in (y, y_hat, selectors)
        # corresponds to "name".
        if n_props > 1:
            data_subset = [j for j, x in enumerate(selectors) if x[ind] != 0.0]
        else:
            data_subset = list(range(len(y)))
        if len(data_subset) > 0:
            y_subset = y[data_subset]
            y_hat_subset = y_hat[data_subset]
            if inverse_transform:
                scalers[name].inverse_transform(y_subset)
                scalers[name].inverse_transform(y_hat_subset)
            return_dict[name] = metric_fn(y_subset, y_hat_subset, mt=False)
        else:
            # If there are no samples corresponding to name then the
            # error metrics must be nan.
            return_dict[name] = nan, nan

    return return_dict


def analyze_gradients(named_parameters, allow_errors=False):
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            try:
                ave_grad = cpu_detach(p.grad.abs().mean().flatten())
                max_grad = cpu_detach(p.grad.abs().max().flatten())
                ave_grads.extend(ave_grad.numpy().tolist())
                max_grads.extend(max_grad.numpy().tolist())
                layers.append(n)
            except:
                print(n)
                print(p.grad)
                if not allow_errors:
                    raise BaseException
    ave_grads = np.array(ave_grads)
    max_grads = np.array(max_grads)
    print("\n..Ave_grads: ", list(zip(layers, ave_grads)))
    return layers, ave_grads, max_grads


def weight_reset(layer):
    if hasattr(layer, "reset_parameters"):
        layer.reset_parameters()


def cpu_detach(tens):

    return tens.detach().cpu()


def get_unit_sequence(input_dim, output_dim, n_hidden):
    """
    Smoothly decay the number of hidden units in each layer.
    Start from 'input_dim' and end with 'output_dim'.

    Examples:
    get_unit_sequence(32, 8, 3) = [32, 16, 16, 16, 3]
    """

    decrement = lambda x: 2 ** (x // 2 - 1).bit_length()
    sequence = [input_dim]
    for _ in range(n_hidden):
        last_num_units = sequence[-1]
        power2 = decrement(last_num_units)
        if power2 > output_dim:
            sequence.append(power2)
        else:
            sequence.append(last_num_units)
    sequence.append(output_dim)

    return sequence


def sorted_attrs(obj):
    """
    Get back sorted attributes of obj. All methods starting with '__' are filtered out.
    """
    return sorted(
        [
            (a, v)
            for a, v in obj.__dict__.items()
            if not a.startswith("__") and not callable(getattr(obj, a))
        ],
        key=lambda x: x[0],
    )


def module_name(obj):
    klass = obj.__class__
    module = klass.__module__

    return module


def get_input_dim(data):
    """
    Get the input dimension of your PyTorch Data
    Keyword arguments:
        data (torch_geometric.data.Data)
    """
    dim = 0
    for name in ks._F_SET:
        if hasattr(data, name):
            tens = getattr(data, name)
            if torch.numel(tens) == 0:
                dim += 0
            else:
                dim += tens.shape[1]
    return dim


def get_output_dim(data):
    """
    Get the output dimension of your PyTorch Data
    Keyword arguments:
        data (torch_geometric.data.Data)
    """
    warnings.warn("This function has been deprecated.")
    return torch.numel(getattr(data, ks._Y))


class GpuWait:
    """
    Keyword arguments:
        patience: The number of seconds between when we search GPUs
        max_wait: The max number of seconds to wait
        max_load: Maximum current relative load for a GPU to be considered available. GPUs with a load larger than max_load is not returned.
    """

    def __init__(self, patience, max_wait, max_load) -> None:
        self.patience = patience
        self.max_wait = max_wait
        self.max_load = max_load
        self.min_load = 0.06
        self.wait = True

    def __enter__(self):
        max_load = max(self.max_load, self.min_load)
        start = datetime.now()
        dt_string = start.strftime("%d/%m/%Y %H:%M:%S")
        print(f"\nGPU wait started at {dt_string}", flush=True)
        while self.wait:
            availableGPUs = GPUtil.getAvailable(
                order="first",
                limit=1,
                maxLoad=max_load,
                maxMemory=max_load,
                includeNan=False,
                excludeID=[],
                excludeUUID=[],
            )
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            if availableGPUs:
                self.wait = False
                print(f"GPU checked at {dt_string}. Free.", flush=True)
            else:
                print(f"GPU checked at {dt_string}. Still busy.", flush=True)
                time.sleep(self.patience)
            time_delta = (now - start).total_seconds()
            if time_delta > self.max_wait:
                print(
                    f"Function terminated. GPU has not been free for {time_delta} seconds while the max_wait was set to {self.max_wait}.",
                    flush=True,
                )
                break

        if not self.wait:
            self.exec_start = datetime.now()
            print(
                f'Executable started at {self.exec_start.strftime("%d/%m/%Y %H:%M:%S")}.',
                flush=True,
            )

        return self.wait

    def __exit__(self, type, value, traceback):
        if not self.wait:
            exec_end = datetime.now()
            print(
                f'Executable finished at {exec_end.strftime("%d/%m/%Y %H:%M:%S")}.',
                flush=True,
            )
            time_delta = (exec_end - self.exec_start).total_seconds()
            print(f"Executable took {time_delta} seconds to run.", flush=True)


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def _assemble_data(model, data):
    if hasattr(model, "assemble_data"):
        x = model.assemble_data(data)
    else:
        x = data
    return x


def lazy_property(fn):
    """
    Implementation borrowed from https://towardsdatascience.com/what-is-lazy-evaluation-in-python-9efb1d3bfed0
    """
    attr_name = "_lazy_" + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


def get_selector_dim(data):
    """
    Keyword args
        data (torch_geometic.data.Data)
    """
    selector_dim = None
    if hasattr(data, ks._F_SELECTORS):
        if getattr(data, ks._F_SELECTORS) != None:
            selector_dim = getattr(data, ks._F_SELECTORS).size()[-1]

    return selector_dim


def weight_classes(label_frequency):
    """
    Return a numpy array of weights for each class label. The less frequent
    the class, the higher its weight.

    Keyword arguments
        class_labels (dict): Dictionary where keys will be class labels
            (e.g., 0) and the values will be an integer frequency (e.g., 12).
    """
    n_classes = len(label_frequency)
    assert list(range(n_classes)) == list(sorted(label_frequency.keys()))
    result = sorted(
        [(k, v) for k, v in label_frequency.items()],
        key=lambda x: x[1],
        reverse=False,
    )
    sorted_freq = sorted([x[1] for x in result], reverse=True)
    result = [k for k, _ in result]
    result = {label: sorted_freq[ind] for ind, label in enumerate(result)}
    result = np.array([result[label] for label in range(n_classes)])
    _min = np.min(sorted_freq)
    result = result / _min
    return result
