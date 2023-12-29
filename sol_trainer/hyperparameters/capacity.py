import time
from collections import deque
import warnings
from copy import deepcopy
import numpy as np
from torch_geometric.loader import DataLoader
from sol_trainer.hyperparameters.hyperparameters import get_optimizer

from sol_trainer.train import (
    amp_train,
    compute_regression_metrics,
    compute_classification_metrics,
    initialize_training,
)
from sol_trainer.utils import get_selector_dim


# #########################
# Code related to capacity
# #########################
DL_DBG_CMS_NSHOW = 5
DL_DBG_SUFFICIENT_R2_ALL_DATA = 0.97
DL_DBG_SUFFICIENT_F1_ALL_DATA = 0.97


def default_per_epoch_trainer(
    epoch,
    train_loader,
    model,
    optimizer,
    selector_dim,
    train_config,
    start,
):
    y = []
    y_hat = []
    error_dict = {
        "r2": None,
        "rmse": None,
        "f1": None,
        "acc": None,
    }
    for _, data in enumerate(train_loader):  # loop through training batches
        data = data.to(train_config.device)
        data = amp_train(model, data, optimizer, train_config, selector_dim)
        y += data.y.detach().cpu().numpy().tolist()
        y_hat += data.yhat.detach().cpu().numpy().tolist()
    y = np.array(y)
    y_hat = np.array(y_hat)
    print("\n....Epoch %s" % epoch)
    if train_config.regression:
        rmse, r2 = compute_regression_metrics(y, y_hat, False)
        error_dict["rmse"], error_dict["r2"] = rmse, r2
        print(f"......[rmse] {rmse} [r2] {r2}")
    else:
        acc, f1 = compute_classification_metrics(y, y_hat, False)
        error_dict["acc"], error_dict["f1"] = acc, f1
        print(f"......[acc] {acc} [f1] {f1}")
    print("......Outputs", y_hat[0:DL_DBG_CMS_NSHOW])
    print("......Labels ", y[0:DL_DBG_CMS_NSHOW])
    end = time.time()
    print(f"......Total time til this epoch {end-start}")

    return error_dict


def choose_model_size_by_overfit(
    model_class,
    model_kwargs_dict,
    capacity_ls,
    data_set,
    train_config,
    patience=3,
    delta=0.01,
):
    """
    Return the smallest capacity capable of over-fitting the data or
    the capacity with the highest R2.

    Keyword args
        model_class: The *class* of model to use
        model_kwargs_dict (dict): A dictionary of parameters, other than
            hps, needed to instantiate objects of class model_class
        capacity_ls (list): A list of capacity values to try
        data_set (List[torch_geometic.data.Data]): The data to overfit
        train_config (trainConfig)
    """
    print("\nBeginning model size search", flush=True)

    train_loader = DataLoader(
        data_set,
        batch_size=train_config.hps.batch_size.get_value(),
        shuffle=True,
        drop_last=True,
    )

    min_best_rmse = np.inf
    max_best_r2 = -np.inf
    max_best_acc = -np.inf
    max_best_f1 = -np.inf
    best_model_n = None  # index of best model
    overfit = False  # overfit is set to True inside the loop if we overfit the data
    fedup = False
    _queue = deque(maxlen=patience)
    # below, force optimizer to be adam
    if train_config.hps.optimizer.get_value() != "adam":
        train_config.hps.optimizer.set_value("adam")
        warnings.warn("Forcing optimizer to be adam.")
    # below, force dropout to be zero
    if train_config.hps.dropout_pct.get_value() != 0.0:
        train_config.hps.dropout_pct.set_value(0.0)
        warnings.warn("Forcing dropout_pct to be zero.")
    selector_dim = get_selector_dim(data_set[0])
    for model_n, capacity in enumerate(capacity_ls):
        if overfit:
            break
        print(
            f"\n..Training model {model_n + 1} (with capacity {capacity}) of {len(capacity_ls)}"
        )
        temp_hps = deepcopy(train_config.hps)
        temp_hps.capacity.set_value(capacity)
        if "hps" not in model_kwargs_dict:
            model_kwargs_dict["hps"] = temp_hps
        model = model_class(**model_kwargs_dict)
        model = model.to(train_config.device)
        model.train()
        optimizer = get_optimizer(model, temp_hps)
        min_rmse = np.inf  # epoch-wise loss
        max_r2 = -np.inf
        max_f1 = -np.inf
        max_acc = -np.inf
        start = time.time()
        for epoch in range(train_config.epochs):
            if overfit:
                break
            train_config.curr_epoch = epoch
            error_dict = default_per_epoch_trainer(
                epoch=epoch,
                train_loader=train_loader,
                model=model,
                optimizer=optimizer,
                selector_dim=selector_dim,
                train_config=train_config,
                start=start,
            )
            if train_config.regression:
                r2 = error_dict["r2"]
                rmse = error_dict["rmse"]
                if r2 > max_r2:
                    # OK, if we reached here then that means the
                    # updates seen during this epoch yield the
                    # best predictions.
                    min_rmse = rmse
                    max_r2 = r2
                if (
                    max_r2 > DL_DBG_SUFFICIENT_R2_ALL_DATA
                ):  # exit model testing loop if this model did good enough
                    best_model_n = model_n
                    print(
                        "Data was overfit with capacity %s" % capacity_ls[best_model_n],
                        flush=True,
                    )
                    overfit = True
                # if the r2 is too negative then we should re-start training to avoid NaN.
                if r2 < -(10**8):
                    model, optimizer = initialize_training(
                        model, train_config.hps, train_config.device
                    )
                print(
                    "......[best rmse] %s [best r2] %s" % (min_rmse, max_r2), flush=True
                )
            else:
                acc = error_dict["acc"]
                f1 = error_dict["f1"]
                if f1 > max_f1:
                    # OK, if we reached here then that means the
                    # updates seen during this epoch yield the
                    # best predictions.
                    max_f1 = f1
                    max_acc = acc
                if (
                    max_f1 > DL_DBG_SUFFICIENT_F1_ALL_DATA
                ):  # exit model testing loop if this model did good enough
                    best_model_n = model_n
                    print(
                        "Data was overfit with capacity %s" % capacity_ls[best_model_n],
                        flush=True,
                    )
                    overfit = True
                print(
                    "......[best acc] %s [best f1] %s" % (max_acc, max_f1), flush=True
                )

            # end of one epoch
        if overfit:
            break
        if train_config.regression:
            _queue.append(max_r2)
        else:
            _queue.append(max_f1)
        fedup = parse_queue(_queue, delta)
        if fedup:
            best_model_n = model_n - patience + 1
            print(
                f"The performance of models with capacity {capacity_ls[best_model_n+1]} through {capacity_ls[model_n]} was not sufficiently better than the performance of the model with capacity {capacity_ls[best_model_n]}."
            )
            break
        # OK, we did not do good enough to overfit the data. But,
        # if this capacity was the best we have seen yet, take note.
        if train_config.regression:
            if min_rmse < min_best_rmse:
                min_best_rmse, max_best_r2, best_model_n = min_rmse, max_r2, model_n
        else:
            if max_f1 > max_best_f1:
                max_best_f1, max_best_acc, best_model_n = max_f1, max_acc, model_n
        # end of loop over all epochs

    print(
        "\nFinished model size search. The optimal capacity is %s.\n"
        % capacity_ls[best_model_n],
        flush=True,
    )
    if train_config.regression:
        return capacity_ls[best_model_n], min_best_rmse
    else:
        return capacity_ls[best_model_n], max_best_f1


def parse_queue(que, delta):
    if len(que) == que.maxlen:
        r2_small_capacity = que[0]
        r2_big_capacity = max(list(que)[1:])
        if (r2_big_capacity - r2_small_capacity) < delta:
            return True

    return False
