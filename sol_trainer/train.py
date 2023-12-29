from torch import nn, manual_seed, cuda, Tensor, tensor
from torch.cuda import amp
from torch import save as torch_save
from torch import device as torch_device
import numpy as np
from torch_geometric.loader import DataLoader
from collections import deque
import random
from os.path import join as join, exists
from sklearn.model_selection import KFold
from dataclasses import dataclass
import warnings

from . import constants as ks
from .utils import (
    _assemble_data,
    get_selector_dim,
    weight_reset,
    analyze_gradients,
    compute_regression_metrics,
    compute_classification_metrics,
    print_per_property_metrics,
)
from sol_trainer import hyperparameters
from sol_trainer.layers import identity
from .infer import eval_submodel
from .scale import *
from . import save
from sol_trainer import prepare, __version__, infer

# fix random seed
random.seed(2)
manual_seed(2)
np.random.seed(2)


def initialize_training(model, hps, device):
    """
    Initialize a model and optimizer for training using just the model's class
    """
    # deal with optimizer
    optimizer = hyperparameters.get_optimizer(model, hps)
    # deal with model
    # implementation modified from https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819
    model.apply(weight_reset)
    model = model.to(device)
    model.train()
    if hasattr(model, "submodel_dict"):
        infer._model_eval_mode(model.submodel_dict, dropout_mode="test")

    return model, optimizer


def amp_train(model, data, optimizer, tc, selector_dim):
    """
    This function handles the parts of the per-epoch loop that torch's
    autocast methods can speed up. See https://pytorch.org/docs/1.9.1/notes/amp_examples.html
    """
    optimizer.zero_grad()
    if tc.amp:
        with amp.autocast(enabled=True):
            x = _assemble_data(model, data)
            if tc.multi_head:
                data = model(x).view(data.num_graphs, selector_dim)
            else:
                data = model(x)
            loss = tc.loss_obj(data)
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            tc.scaler.scale(loss).backward()

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            tc.scaler.step(optimizer)
            hyperparameters.update_optimizer(
                optimizer, tc.hps, tc.epochs, tc.curr_epoch
            )

            # Updates the scale for next iteration.
            tc.scaler.update()
    else:
        data, loss = minibatch(data, tc, model, selector_dim)
        loss.backward()
        optimizer.step()
        hyperparameters.update_optimizer(optimizer, tc.hps, tc.epochs, tc.curr_epoch)

    # the only thing we need to return is "output". Both "model" and
    # "optimizer" get updated. But these updates happen in place.
    return data


def minibatch(data, tc, model, selector_dim):
    x = _assemble_data(model, data)
    if tc.multi_head:
        data = model(x).view(data.num_graphs, selector_dim)
    else:
        data = model(x)
    loss = tc.loss_obj(data)

    return data, loss


def train_submodel(
    model,
    train_pts,
    val_pts,
    scalers,
    tc,  # train_config
    break_bad_grads=True,
):
    """
    Keyword arguments:
        model (nn.Module)
        train_pts (list)
        val_pts (list)
        scalers (dict)
        tc (trainConfig)
        break_bad_grads (bool):  If True, we will exit the training loop
            after noticing exploding/vanishing gradients early in training.
            If False, we will re-initialize the model after noticing
            exploding/vanishing gradients early in training.
    """

    # error handle inputs
    if tc.model_save_path:
        if not tc.model_save_path.endswith(".pt"):
            raise ValueError(f"The model_save_path you passed in does not end in .pt")
    model.to(tc.device)
    optimizer = hyperparameters.get_optimizer(model, tc.hps)

    error_dict, train_loader = train_all_epochs(
        model,
        train_pts,
        val_pts,
        optimizer,
        scalers,
        tc,
        break_bad_grads,
    )
    train_end(model, train_loader, optimizer, tc)
    if tc.regression:
        result = error_dict["min_val_rmse"]
    else:
        # For now, let's return the negative of the f1 score since
        # the current function is often called by some MINIMIZATION
        # function, e.g. "skopt.gp_minimize".
        result = -error_dict["max_val_f1"]
    return result


def train_all_epochs(
    model,
    train_pts,
    val_pts,
    optimizer,
    scalers,
    train_config,
    break_bad_grads,
):
    """
    Train over all the epochs specified in train_config
    """
    # to determine the presence of a "selector_dim" the first member
    # of val_X should have an attribute named selector and the value of
    # the attribute should not be None
    selector_dim = get_selector_dim(train_pts[0])
    # if we do not need to make a new dataloader inside each epoch,
    # let us make the dataloader now.
    if not train_config.get_train_pts:
        train_loader = DataLoader(
            train_pts, batch_size=train_config.hps.batch_size.value, shuffle=True
        )
    val_loader = DataLoader(
        val_pts, batch_size=train_config.hps.batch_size.value * 2, shuffle=True
    )
    # create the epoch suffix for this submodel
    epoch_suffix = f"{train_config.epoch_suffix}, fold {train_config.fold_index}"
    # intialize a few variables that get reset during the training loop
    min_val_rmse = np.inf  # epoch-wise loss
    max_val_r2 = -np.inf
    max_val_acc = -np.inf
    max_val_f1 = -np.inf
    best_val_epoch = 0
    vanishing_grads = False
    exploding_grads = False
    exploding_errors = False
    train_config.grad_hist_per_epoch = deque(
        maxlen=ks.GRADIENT_HISTORY_LENGTH
    )  # gradients for last maxlen epochs
    for epoch in range(train_config.epochs):
        train_config.curr_epoch = epoch
        # Let's stop training and not waste time if we have vanishing
        # gradients early in training. We won't
        # be able to learn anything anyway.
        if vanishing_grads:
            print("Vanishing gradients detected")
        if exploding_errors:
            print("Exploding errors detected")
        if exploding_grads:
            print("Exploding gradients detected")
        if (
            (vanishing_grads or exploding_grads)
            and (train_config.curr_epoch < 50)
            and break_bad_grads
        ):
            break
        # If the errors or gradients are messed up later in training,
        # let us just re-initialize the model. Perhaps this new initial
        # point on the loss surface will lead to a better local minima.
        elif exploding_errors or vanishing_grads or exploding_grads:
            model, optimizer = initialize_training(
                model, train_config.hps, train_config.device
            )
        # augment data, if necessary
        if train_config.get_train_pts:
            train_pts = train_config.get_train_pts()
            train_loader = DataLoader(
                train_pts, batch_size=train_config.hps.batch_size.value, shuffle=True
            )
        # #####################################################################
        # Enter and execute the training loop for the current epoch.
        # #####################################################################
        y = []
        y_hat = []
        selectors = []
        model.train()
        if hasattr(model, "submodel_dict"):
            infer._model_eval_mode(model.submodel_dict, dropout_mode="test")
        for ind, data in enumerate(train_loader):  # loop through training batches
            data = data.to(train_config.device)
            data = amp_train(model, data, optimizer, train_config, selector_dim)
            y += data.y.cpu().numpy().tolist()
            y_hat += data.yhat.detach().cpu().numpy().tolist()
            if selector_dim:
                selectors += data.selector.cpu().numpy().tolist()

        y = np.array(y)
        y_hat = np.array(y_hat)
        _, ave_grads, _ = analyze_gradients(
            model.named_parameters(), allow_errors=False
        )
        train_config.grad_hist_per_epoch.append(ave_grads)
        if train_config.regression:
            # rmse on data in loss function scale
            tr_rmse, tr_r2 = compute_regression_metrics(
                y, y_hat, train_config.multi_head
            )
            # check for exploding errors, vanishing grads, and exploding grads
            if tr_r2 < ks.DL_STOP_TRAIN_R2:
                exploding_errors = True
            else:
                exploding_errors = False
        else:
            tr_acc, tr_f1 = compute_classification_metrics(
                y, y_hat, train_config.multi_head
            )
        if np.sum(train_config.grad_hist_per_epoch) == 0:
            vanishing_grads = True
        else:
            vanishing_grads = False
        if int(np.sum(np.isnan(train_config.grad_hist_per_epoch))) == len(
            train_config.grad_hist_per_epoch
        ):
            exploding_grads = True
        else:
            exploding_grads = False
        # #####################################################################

        # Below, we execute the validation loop for the current epoch.
        print("selector_dim", selector_dim)
        y_val, y_val_hat, selectors_val = eval_submodel(
            model, val_loader, train_config.device, selector_dim
        )
        print(f"\nEpoch {epoch}{epoch_suffix}", flush=True)
        # #########################################################################################################
        # Compute then print overall error metrics.
        # #########################################################################################################
        if train_config.regression:
            val_rmse, val_r2 = compute_regression_metrics(
                y_val, y_val_hat, train_config.multi_head
            )
            print(
                "[loss scale val rmse] %s [loss scale val r2] %s [loss scale tr rmse] %s [loss scale tr r2] %s"
                % (val_rmse, val_r2, tr_rmse, tr_r2),
                flush=True,
            )
        else:
            val_acc, val_f1 = compute_classification_metrics(
                y_val, y_val_hat, train_config.multi_head
            )
            print(
                "[loss scale val acc] %s [loss scale val f1] %s [loss scale tr acc] %s [loss scale tr f1] %s"
                % (val_acc, val_f1, tr_acc, tr_f1),
                flush=True,
            )
        print_per_property_metrics(
            y_val,
            y_val_hat,
            selectors_val,
            scalers,
            inverse_transform=True,
            regression=train_config.regression,
        )
        # #########################################################################################################

        # ####################################################################################
        # Checkpoint the model if its the best we have seen so far.
        # Then, print out that the model has been saved. Then, update the error metrics.
        # Finally, print out error metrics for the best model seen so far.
        # ####################################################################################
        if train_config.regression:
            if val_rmse < min_val_rmse:
                min_val_rmse = val_rmse
                max_val_r2 = val_r2
                best_val_epoch = epoch
                if train_config.model_save_path:
                    torch_save(model.state_dict(), train_config.model_save_path)
                    print("Best model saved", flush=True)
            print(
                "[best val epoch] %s [best val loss scale rmse] %s [best val loss scale r2] %s"
                % (best_val_epoch, min_val_rmse, max_val_r2),
                flush=True,
            )
        else:
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
                max_val_acc = val_acc
                best_val_epoch = epoch
                if train_config.model_save_path:
                    torch_save(model.state_dict(), train_config.model_save_path)
                    print("Best model saved", flush=True)
            print(
                "[best val epoch] %s [best val acc] %s [best val f1] %s"
                % (best_val_epoch, max_val_acc, max_val_f1),
                flush=True,
            )
        # ####################################################################################
    error_dict = {
        "min_val_rmse": min_val_rmse,
        "max_val_f1": max_val_f1,
    }
    return error_dict, train_loader


def train_end(model, train_loader, optimizer, train_config):
    if hyperparameters.is_SWA(optimizer):
        optimizer.swap_swa_sgd()
        # if we are doing normalization then we need to reset all the
        # moving averages after SWA
        if train_config.hps.norm.get_value() != identity:
            for _, data in enumerate(train_loader):  # loop through training batches
                data = data.to(train_config.device)
                _ = model(_assemble_data(model, data))


def train_kfold_ensemble(
    dataframe,
    submodel_cls,
    submodel_kwargs_dict,
    train_config,
    submodel_trainer,
    augmented_featurizer,
    scaler_dict,
    root_dir,
    n_fold,
    random_seed,
):
    """
    Train an ensemble model on dataframe.

    Keyword arguments:
        dataframe (pd.DataFrame): A dataframe consisting of atleast two columns:
            value, data. smiles_string is an optional column. This dataframe
            should not contain any na.
        model_constructor: A lambda function that returns an nn.Module object
            when called.
        train_config (trainConfig)
        submodel_trainer (callable): A function to train the output of
            model_constructor
        augmented_featurizer: A function that takes in a smiles string,
            augments it, and returns its features.
        scaler_dict (dict):
        root_dir (str): The path to directory where all data will be
            saved. This string should match the string used in
            prepare_train.
        n_fold (int): The number of folds to use during training
        random_seed (int)
    """
    # The root directories and subdirectories should have already been
    # created by prepare_train. Let us retrieve the path to the
    # subdirectories so that we can save some more stuff
    model_dir, md_dir = save.get_root_subdirs(root_dir)
    # Check if we are doing regression or classification, and assign the
    # result to the train_config.
    if exists(join(md_dir, ks.CLASSLABELS_FILENAME)):
        regression = False
    else:
        regression = True
    train_config.regression = regression
    # save scaler_dict
    save.safe_save(scaler_dict, join(md_dir + ks.SCALERS_FILENAME), "pickle")
    # make selector dict and save it
    prop_cols = sorted(list(scaler_dict.keys()))
    selector_dict = {}
    for prop in prop_cols:
        selector = dataframe[dataframe.prop == prop].selector.values.tolist()[0]
        if not isinstance(selector, Tensor):
            selector = tensor(selector, requires_grad=False)
        selector_dict[prop] = selector
    save.safe_save(selector_dict, join(md_dir + ks.SELECTORS_FILENAME), "pickle")
    # save property name list
    save.safe_save("\n".join(prop_cols), join(md_dir + ks.PROP_FILENAME), "text")
    # save hyperparams
    save.safe_save(train_config.hps, join(md_dir + ks.HPS_FILENAME), "pickle")
    # ######################################
    # helper functions for CPU-based data augmentation
    # ######################################
    def get_data_augmented(x):
        """
        Return a Data object with the augmented smiles. This function
        requires that dataframe contains a column named data
        """
        data = augmented_featurizer(x.smiles_string)
        data.y = x.data.y  # copy label directly from x.data as that label
        # should already be scaled.
        prepare.copy_attribute_safe(x.data, data, "selector")
        prepare.copy_attribute_safe(x.data, data, "graph_feats")
        prepare.copy_attribute_safe(x.data, data, "node_feats")

        return data

    def cv_get_train_pts(training_df):
        """
        Return a list of augmented Data objects
        """

        return training_df.apply(get_data_augmented, axis=1).values.tolist()

    # #########################################
    # do cross-validation
    kf_ = KFold(
        n_splits=n_fold,
        shuffle=True,
        random_state=random_seed,
    )
    kf = kf_.split(range(len(dataframe)))
    ind = 0
    for train, val in kf:
        # train submodel
        print(f"Fold {ind}: training inds are ... {train}")
        print(f"Fold {ind}: validation inds are ... {val}")
        train_config.model_save_path = join(model_dir, f"model_{ind}.pt")
        training_df = dataframe.iloc[train, :]
        val_df = dataframe.iloc[val, :]
        val_pts = val_df["data"].values.tolist()
        train_pts = training_df["data"].values.tolist()
        if augmented_featurizer:
            train_config.get_train_pts = lambda: cv_get_train_pts(training_df)
        model = submodel_cls(**submodel_kwargs_dict)
        train_config.fold_index = ind  # add the fold index to train_config
        submodel_trainer(
            model, train_pts, val_pts, scaler_dict, train_config, break_bad_grads=False
        )
        ind += 1


@dataclass
class trainConfig:
    """
    A class to pass into the submodel trainer
    """

    # need to be set manually
    loss_obj: nn.Module
    amp: bool  # when using T2 this should be set to False
    # hps: HpConfig = None
    # loss_obj: nn.Module = None

    # ##############################################
    # The params below are given default values.
    # ##############################################
    # By default, set regression to True to ensure backward
    # compatibility.
    regression: bool = None
    device: torch_device = torch_device(
        "cuda" if cuda.is_available() else "cpu"
    )  # specify GPU
    epoch_suffix: str = ""
    multi_head: bool = None
    # ###############################################

    # set dynamically inside train_kfold_ensemble, so
    # we can set each attribute to None on instantiation
    hps: hyperparameters.HpConfig = None
    model_save_path: str = None
    fold_index = None
    get_train_pts = None

    def __post_init__(self):
        if self.amp:
            self.scaler = amp.GradScaler()
        if self.regression == None:
            self.regression = True
            warnings.warn(
                "A value for the 'regression' attribute was not passed "
                f"in. Therefore, it is assumed to be {self.regression}."
            )


def prepare_train(
    dataframe,
    smiles_featurizer,
):
    """
    An alias to prepare.prepare_train
    """
    return prepare.prepare_train(
        dataframe,
        smiles_featurizer,
    )


def train_kfold_LinearEnsemble(
    dataframe,
    submodel_cls,
    submodel_kwargs_dict,
    train_config,
    submodel_trainer,
    augmented_featurizer,
    scaler_dict,
    root_dir,
    n_fold,
    random_seed,
):
    """
    Train a LinearEnsemble on dataframe.

    Keyword arguments:
        dataframe (pd.DataFrame): A dataframe consisting of atleast two columns:
            value, data. smiles_string is an optional column. This dataframe
            should not contain any na.
        model_constructor: A lambda function that returns an nn.Module object
            when called.
        train_config (trainConfig)
        submodel_trainer (callable): A function to train the output of
            model_constructor
        augmented_featurizer: A function that takes in a smiles string,
            augments it, and returns its features.
        scaler_dict (dict):
        root_dir (str): The path to directory where all data will be
            saved. This string should match the string used in
            prepare_train.
        n_fold (int): The number of folds to use during training
        random_seed (int)
    """
    return train_kfold_ensemble(
        dataframe,
        submodel_cls,
        submodel_kwargs_dict,
        train_config,
        submodel_trainer,
        augmented_featurizer,
        scaler_dict,
        root_dir,
        n_fold,
        random_seed,
    )


def train_kfold_submodels(
    dataframe,
    submodel_cls,
    submodel_kwargs_dict,
    train_config,
    submodel_trainer,
    augmented_featurizer,
    scaler_dict,
    root_dir,
    n_fold,
    random_seed,
):
    """
    Train submodels to be used later for a metamodel

    Keyword arguments:
        dataframe (pd.DataFrame): A dataframe consisting of atleast two columns:
            value, data. smiles_string is an optional column. This dataframe
            should not contain any na.
        train_config (trainConfig)
        submodel_trainer (callable): A function to train the output of
            model_constructor
        augmented_featurizer: A function that takes in a smiles string,
            augments it, and returns its features.
        scaler_dict (dict):
        root_dir (str): The path to directory where all data will be
            saved. This string should match the string used in
            prepare_train.
        n_fold (int): The number of folds to use during training
        random_seed (int)
    """
    return train_kfold_ensemble(
        dataframe,
        submodel_cls,
        submodel_kwargs_dict,
        train_config,
        submodel_trainer,
        augmented_featurizer,
        scaler_dict,
        root_dir,
        n_fold,
        random_seed,
    )
