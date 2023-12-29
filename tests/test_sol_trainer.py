import pytest
from torch import manual_seed, nn
import os
import shutil
import pandas as pd
from skopt import gp_minimize
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from copy import deepcopy
import warnings
import time

from sol_trainer import __version__
from sol_trainer import save, loss, constants, models, load
from sol_trainer import utils as st_utils
from sol_trainer.hyperparameters import (
    HpConfig,
    ModelParameter,
    choose_model_size_by_overfit,
    compute_batch_size_range,
    get_optimizer,
    update_optimizer,
)
from sol_trainer.layers import (
    LEAKY_RELU_DEFAULT_SLOPE,
    initialize_weights,
)
from sol_trainer.infer import eval_ensemble
from sol_trainer.layers import UOut, identity, my_hidden
from sol_trainer.train import (
    train_all_epochs,
    train_end,
    train_kfold_LinearEnsemble,
    train_kfold_ensemble,
    train_submodel,
    trainConfig,
)
from sol_trainer.prepare import check_series_values, prepare_train
from .utils_test import (
    MathModel1,
    copy_params,
    get_prelu_slopes,
    has_bn,
    trainer_MathModel,
    morgan_featurizer,
    ensemble_trainer_helper,
)

# set seeds for reproducibility
random.seed(12)
manual_seed(12)
np.random.seed(12)


def test_version():
    assert __version__ == "0.1.0"


@pytest.fixture
def example_data():
    properties = ["property1"] * 7 + ["property2"] * 7
    values = np.random.randn(
        14,
    )  # random data
    smiles = [
        "[*]CC[*]",
        "[*]CC(C)[*]",
        "[*]CCN[*]",
        "[*]CCO[*]",
        "[*]CCCN[*]",
        "[*]C(O)C[*]",
        "[*]C(CCC)C[*]",
    ] * 2
    data = {
        "prop": properties,
        "value": values,
        "smiles_string": smiles,
        "graph_feats": [{} for _ in smiles],
        "node_feats": [{} for _ in smiles],
    }
    dataframe = pd.DataFrame(data)

    base_train_config = trainConfig(
        loss_obj=loss.sh_mse_loss(),
        amp=False,  # False since we are running a test on a CPU
    )
    # some attributes need to be defined AFTER instantiation since
    # __init__ does not know about them
    base_train_config.device = torch.device("cpu")
    base_train_config.do_augment = False
    base_train_config.multi_head = False
    # tc_ensemble.loss_obj = loss.sh_mse_loss()

    hp_space = [
        (np.log10(0.0003), np.log10(0.03)),  # learning rate
        (1, 10),  # batch size
        (0, 0.5),  # dropout
        (2, 4),  # capacity
    ]

    return {
        "dataframe": dataframe,
        "properties": properties,
        "base_train_config": base_train_config,
        "hp_space": hp_space,
    }


@pytest.fixture
def example_mt_regression_data(example_data):
    base_train_config = example_data["base_train_config"]
    root = (
        "mt_regression_ensemble/",
    )  # root directory for ensemble trained on MT data
    dataframe = deepcopy(example_data["dataframe"])
    return {
        "root": root,
        "dataframe": dataframe,
        "base_train_config": base_train_config,
    }


@pytest.fixture
def example_st_regression_data(example_data, example_mt_regression_data):
    root = (
        "st_regression_ensemble/",
    )  # root directory for ensemble trained on ST data
    dataframe = deepcopy(example_data["dataframe"])
    # Let's remove "property2" data from the dataframe so we get a
    # single-task model later.
    dataframe = dataframe[dataframe.prop == "property1"]
    return {
        "root": root,
        "dataframe": dataframe,
        "base_train_config": example_mt_regression_data["base_train_config"],
    }


@pytest.fixture
def example_st_classification_data(example_st_regression_data):
    root = (
        "st_classification_ensemble/",
    )  # root directory for ensemble trained on ST data
    dataframe = deepcopy(example_st_regression_data["dataframe"])
    # Convert dataframe["value"] from continuous to data to discrete data.
    n_classes = 3
    classes = [f"class{i}" for i in range(n_classes)]
    dataframe["value"] = random.choices(classes, k=len(dataframe))
    base_train_config = example_st_regression_data["base_train_config"]
    base_train_config.loss_obj = loss.sh_crossentropy_loss(n_classes)
    base_train_config.regression = False
    #
    return {
        "root": root,
        "dataframe": dataframe,
        "base_train_config": base_train_config,
    }


def test_safe_save_pickle(example_data):
    # save something the first time. This should go without a hitch.
    obj = example_data["properties"]
    save.safe_save(obj, "properties.pkl", "pickle")
    with pytest.raises(ValueError):
        # try saving the same thing again
        save.safe_save(obj, "properties.pkl", "pickle")


# a mark so that tests are run on both single-task and multi-task data
@pytest.mark.parametrize(
    "fixture",
    [
        "example_st_classification_data",
        "example_st_regression_data",
        "example_mt_regression_data",
    ],
)
def test_ensemble_trainer(fixture, request, example_data, capsys):
    """
    This function tests the following.
        1) Can we can run train_kfold_ensemble after hyperparameter
            optimization without error?
        2) Can we use parallelization in prepare_train?
    """
    data_for_test = request.getfixturevalue(fixture)
    regression = data_for_test["base_train_config"].regression
    root = data_for_test["root"]
    dataframe, scaler_dict = prepare_train(
        data_for_test["dataframe"],
        morgan_featurizer,
        root_dir=root,
        ncore=2,  # >1, to activate parallelization.
    )
    assert dataframe.graph_feats.values[0] == {}
    training_df, val_df = train_test_split(
        dataframe,
        test_size=constants.VAL_FRAC,
        stratify=dataframe.prop,
        random_state=constants.RANDOM_SEED,
    )
    train_pts, val_pts = training_df.data.values.tolist(), val_df.data.values.tolist()
    epochs = 2
    # create hyperparameter space
    hp_space = example_data["hp_space"]
    input_dim = st_utils.get_input_dim(train_pts[0])
    if regression:
        output_dim = st_utils.get_output_dim(train_pts[0])
    else:
        output_dim = data_for_test["base_train_config"].loss_obj.n_classes
    if regression:
        MlpType = models.MlpRegressor
    else:
        MlpType = models.MlpClassifier
    # create objective function
    def obj_func(x):
        print(f"Testing hps: {x}")
        hps = HpConfig()
        hps.set_values(
            {
                "r_learn": 10 ** x[0],
                "batch_size": x[1],
                "dropout_pct": x[2],
                "capacity": x[3],
                "activation": nn.functional.leaky_relu,
                "initializer": nn.init.kaiming_normal_,
                "norm": identity,
                "optimizer": "adam",
                "swa_start_frac": 0.0,
                "swa_freq": 0,
                "weight_decay": 0.0,
            }
        )

        # trainConfig for the hp search
        tc_search = deepcopy(data_for_test["base_train_config"])
        # Some attributes need to be defined AFTER instantiation since
        # __init__ does not know about them.
        tc_search.hps = hps
        tc_search.epochs = epochs
        model = MlpType(
            input_dim=input_dim,
            output_dim=output_dim,
            hps=hps,
        )
        # Check that biases are initialized correctly.
        assert model.mlp.layers[0].linear.bias.max() == 0.0
        assert model.mlp.layers[0].linear.bias.min() == 0.0
        # Re-make the model, using PReLU instead.
        INIT_SLOPE = 0.25
        hps.set_values(
            {
                "activation": nn.PReLU(init=INIT_SLOPE),
            }
        )
        model = MlpType(
            input_dim=input_dim,
            output_dim=output_dim,
            hps=hps,
        )
        val_rmse = train_submodel(
            model,
            train_pts,
            val_pts,
            scaler_dict,
            tc_search,
        )
        # Check that the learned slope is not equal to the starting slope.
        final_slopes = get_prelu_slopes(model)
        assert len(final_slopes) > 0
        for slope in final_slopes:
            assert slope != INIT_SLOPE
        return val_rmse

    # obtain the optimal point in hp space
    opt_obj = gp_minimize(
        func=obj_func,
        dimensions=hp_space,
        n_calls=10,
        random_state=0,
    )
    # create an HpConfig from the optimal point in hp space
    hps = HpConfig()
    hps.set_values(
        {
            "r_learn": 10 ** opt_obj.x[0],
            "batch_size": opt_obj.x[1],
            "dropout_pct": opt_obj.x[2],
            "capacity": opt_obj.x[3],
            "activation": nn.functional.leaky_relu,
            "initializer": nn.init.kaiming_normal_,
            "norm": identity,
            "optimizer": "adam",
            "swa_start_frac": 0.0,
            "swa_freq": 0,
            "weight_decay": 0.0,
        }
    )

    # create inputs for train_kfold_ensemble
    submodel_kwargs_dict = dict(
        input_dim=input_dim,
        output_dim=output_dim,
        hps=hps,
    )
    dataframe = pd.concat(
        [val_df, training_df],
        ignore_index=True,
    )
    # trainConfig for the ensemble training
    tc_ensemble = deepcopy(data_for_test["base_train_config"])
    # some attributes need to be defined AFTER instantiation since
    # __init__ does not know about them
    tc_ensemble.hps = hps
    tc_ensemble.epochs = epochs
    train_kfold_ensemble(
        dataframe,
        MlpType,
        submodel_kwargs_dict,
        tc_ensemble,
        train_submodel,
        augmented_featurizer=None,  # since we do not want augmentation
        scaler_dict=scaler_dict,
        root_dir=root,
        n_fold=2,
        random_seed=234,
    )
    captured = capsys.readouterr()
    if regression:
        # Check that error metrics in original scale are printed.
        assert "[property1 orig. scale val rmse]" in captured.out
    else:
        # Check that f1 score and accuracy are printed for classification
        # problems.
        assert "[property1 orig. scale val acc]" in captured.out
        assert "[property1 orig. scale val f1]" in captured.out
    if root == "st_regression_ensemble":
        data0 = data_for_test["dataframe"].iloc[:2, :]
        data1 = data0.copy(deep=True)
        data1.index = [1, 0]
        data1 = data1.sort_index()
        assert data0.smiles_string.values[0] == data1.smiles_string.values[1]
        # When using MC dropout, we cannot ensure consistent predictions.
        ensemble = load.load_ensemble(
            models.LinearEnsemble,
            root,
            models.MlpOut,
            "cpu",
            submodel_kwargs_dict,
            {"monte_carlo": True},
        )
        mc_result0, mc_result1 = ensemble_trainer_helper(
            ensemble,
            root,
            data0,
            data1,
        )
        assert mc_result0[0] != mc_result1[1]
        # When NOT using MC dropout, we CAN ensure consistent predictions.
        ensemble = load.load_ensemble(
            models.LinearEnsemble,
            root,
            models.MlpOut,
            "cpu",
            submodel_kwargs_dict,
            {"monte_carlo": False},
        )
        mc_result0, mc_result1 = ensemble_trainer_helper(
            ensemble,
            root,
            data0,
            data1,
        )
        assert mc_result0[0] == mc_result1[1]

    if root == "st_classification_ensemble":
        # Below, let's check that a classification ensemble can infer
        # without error.
        ensemble = load.load_ensemble(
            models.LinearEnsemble,
            root,
            models.MlpOut,
            "cpu",
            submodel_kwargs_dict,
            {"monte_carlo": False},
        )
        eval_ensemble(
            ensemble,
            root,
            data_for_test["dataframe"],
            morgan_featurizer,
            "cpu",
            ensemble_forward_kwargs={"n_passes": 1},
        )

    assert True


# a mark so that tests are run on both single-task and multi-task data
@pytest.mark.parametrize(
    "fixture",
    [
        "example_st_classification_data",
        "example_st_regression_data",
        "example_mt_regression_data",
    ],
)
def test_load_ensemble_noerror(fixture, request):
    """
    This test checks that load_ensemble can be performed without error
    """
    data_for_test = request.getfixturevalue(fixture)
    regression = data_for_test["base_train_config"].regression
    if regression:
        output_dim = 1
    else:
        output_dim = data_for_test["base_train_config"].loss_obj.n_classes
    selectors = load.load_selectors(data_for_test["root"])
    selector_dim = torch.numel(list(selectors.values())[0])
    # calculate input dimension
    input_dim = 512 + selector_dim
    ensemble = load.load_ensemble(
        models.LinearEnsemble,
        data_for_test["root"],
        models.MlpOut,
        device="cpu",
        submodel_kwargs_dict={
            "input_dim": input_dim,
            "output_dim": output_dim,
        },
        ensemble_init_kwargs={},  # This has to be there to avoid a weird PyTest error.
    )
    assert ensemble.regression == regression
    assert len(ensemble.submodel_dict) > 0


@pytest.fixture
def example_unit_sequence():
    hps = HpConfig()
    dictionary = {
        "capacity": 3,
        "dropout_pct": 0.0,
        "activation": nn.functional.leaky_relu,
        "initializer": nn.init.kaiming_normal_,
        "norm": identity,
        "optimizer": "adam",
        "r_learn": 0.0,
        "swa_freq": 0,
        "swa_start_frac": 0.0,
        "weight_decay": 0.0,
    }
    hps.set_values(dictionary)
    return {
        "input_dim": 32,
        "output_dim": 8,
        "hps": hps,
        "capacity": dictionary["capacity"],
        "unit_sequence": [32, 16, 16, 16, 8],
    }


def test_unit_sequence(example_unit_sequence):
    assert (
        st_utils.get_unit_sequence(
            example_unit_sequence["input_dim"],
            example_unit_sequence["output_dim"],
            example_unit_sequence["capacity"],
        )
        == example_unit_sequence["unit_sequence"]
    )


def test_hp_str(example_unit_sequence, capsys):
    # check that hps are printed correctly
    assert (
        str(example_unit_sequence["hps"])
        == "{activation: leaky_relu; batch_size: None; capacity: 3;"
        + " dropout_pct: 0.0; graph_feats_scaler: Forward();"
        + " initializer: kaiming_normal_; node_feats_scaler: Forward();"
        + " norm: identity; optimizer: adam; r_learn: 0.0;"
        + " swa_freq: 0; swa_start_frac: 0.0; weight_decay: 0.0}"
    )
    # ############################################################
    # check that hps are NOT printed when a layer is instantiated
    # ############################################################
    layer = my_hidden(size_in=1, size_out=1, hps=example_unit_sequence["hps"])
    assert (
        st_utils.module_name(layer) == "sol_trainer.layers"
    )  # test that module_name works
    captured = capsys.readouterr()
    check_str = "Hyperparameters after model instantiation"
    assert check_str not in captured.out
    # ############################################################
    # check that hps are printed when a model is instantiated
    # ############################################################
    models.MlpOut(input_dim=1, output_dim=1, hps=example_unit_sequence["hps"])
    captured = capsys.readouterr()
    assert check_str in captured.out


def test_unit_sequence_MlpOut(example_unit_sequence):
    model = models.MlpOut(
        example_unit_sequence["input_dim"],
        example_unit_sequence["output_dim"],
        example_unit_sequence["hps"],
        False,
    )
    assert (
        model.mlp.unit_sequence + [model.output_dim]
        == example_unit_sequence["unit_sequence"]
    )


def test_hp_options():
    """
    Test that Parameter "optimizer" only allows two values, "adam" and "swa"
    """
    hps = HpConfig()
    hps.set_values({"optimizer": "adam"})
    hps.set_values({"optimizer": "swa"})
    with pytest.raises(TypeError, match=f"The value passed in"):
        hps.set_values({"optimizer": "SWA"})
    assert True


def test_hp_norm():
    """
    Test that the norm ModelParameter works
    """
    hps = HpConfig()
    hps.set_values(
        {
            "r_learn": 0.01,
            "batch_size": 10,
            "dropout_pct": 0.1,
            "capacity": 2,
            "activation": nn.functional.leaky_relu,
            "initializer": nn.init.kaiming_normal_,
            "norm": identity,
            "optimizer": "adam",
            "swa_start_frac": 0.0,
            "swa_freq": 0,
            "weight_decay": 0.0,
        }
    )
    model = models.MlpOut(10, 1, hps)
    assert not has_bn(model)
    hps.set_values(
        {
            "norm": nn.BatchNorm1d,
        }
    )
    model = models.MlpOut(10, 1, hps)
    assert has_bn(model)

    # test that the dropout pct is decreased to 0.05
    hps.set_values({"dropout_pct": 0.1})
    model = models.MlpOut(10, 1, hps)
    assert (
        len([m for m in model.modules() if m.__class__.__name__.startswith("Dropout")])
        > 0
    )
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            assert (
                m.p == 0.05
            ), f"Module {str(m)} has dropout {m.p} but it should be 0.05"

    # test that the dropout pct stays at 0.03
    p = 0.03
    hps.set_values({"dropout_pct": p})
    model = models.MlpOut(10, 1, hps)
    assert (
        len([m for m in model.modules() if m.__class__.__name__.startswith("Dropout")])
        > 0
    )
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            assert (
                m.p == 0.03
            ), f"Module {str(m)} has dropout {m.p} but it should be {p}"


def test_update_optimizer():
    # test that no changes to Adam optimizer are made in update_optimizer
    hps = HpConfig()
    hps.set_values(
        {
            "r_learn": 0.01,
            "batch_size": 10,
            "dropout_pct": 0.05,
            "capacity": 1,
            "activation": nn.functional.leaky_relu,
            "initializer": nn.init.kaiming_normal_,
            "norm": identity,
            "optimizer": "adam",
            "swa_start_frac": 0.0,
            "swa_freq": 0,
            "weight_decay": 0.0,
        }
    )
    model = models.MlpOut(10, 1, hps)
    opt = get_optimizer(model, hps)
    start_params = deepcopy(opt.param_groups[0]["params"])
    update_optimizer(opt, hps, 100, 99)
    end_params = deepcopy(opt.param_groups[0]["params"])
    assert all(
        [torch.equal(start, end) for start, end in zip(start_params, end_params)]
    )

    # test that no changes are made to SWA optimizer during initial phase of training
    hps.set_values(
        {
            "optimizer": "swa",
            "swa_start_frac": 0.9,
            "swa_freq": 5,
        }
    )
    model = models.MlpOut(10, 1, hps)
    opt = get_optimizer(model, hps)
    n_avg_start = [group["n_avg"] for group in opt.param_groups]
    update_optimizer(opt, hps, 100, 20)
    n_avg_end = [group["n_avg"] for group in opt.param_groups]

    assert n_avg_start == n_avg_end

    # test that no changes are made to SWA optimizer in cyclical
    # phase of training when curr_epoch is not a multiple of swa_freq
    model = models.MlpOut(10, 1, hps)
    opt = get_optimizer(model, hps)
    n_avg_start = [group["n_avg"] for group in opt.param_groups]
    update_optimizer(opt, hps, 100, 94)
    n_avg_end = [group["n_avg"] for group in opt.param_groups]
    assert n_avg_start == n_avg_end

    # test that changes are made to SWA optimizer in cyclical
    # phase of training when curr_epoch is a multiple of swa_freq
    model = models.MlpOut(10, 1, hps)
    opt = get_optimizer(model, hps)
    n_avg_start = np.array([group["n_avg"] for group in opt.param_groups])
    update_optimizer(opt, hps, 100, 95)
    n_avg_end = np.array([group["n_avg"] for group in opt.param_groups])
    assert np.array_equal(n_avg_start + 1, n_avg_end)


@pytest.fixture
def example_end_data(example_data):
    root = None
    dataframe, scaler_dict = prepare_train(
        example_data["dataframe"], morgan_featurizer, root_dir=root
    )
    return {
        "root": root,
        "dataframe": dataframe,
        "scaler_dict": scaler_dict,
    }


def test_train_end(example_data, example_end_data):

    # Make the HpConfig object.
    hps = HpConfig()
    hps.set_values(
        {
            "r_learn": 0.01,
            "batch_size": 10,
            "dropout_pct": 0.05,
            "capacity": 2,
            "activation": nn.functional.leaky_relu,
            "initializer": nn.init.kaiming_normal_,
            "norm": nn.BatchNorm1d,
            "optimizer": "swa",
            "swa_start_frac": 0.2,
            "swa_freq": 5,
            "weight_decay": 0.0,
        }
    )

    # #####################
    # make the trainConfig
    # #####################
    tc_ensemble = example_data["base_train_config"]
    tc_ensemble.hps = hps
    tc_ensemble.epochs = 10
    # instantiate the model
    input_dim = st_utils.get_input_dim(
        example_end_data["dataframe"].data.values.tolist()[0]
    )
    model = models.MlpOut(input_dim, 1, hps)
    #
    optimizer = get_optimizer(model, hps)
    # check that swap_swa_sgd was performed
    _, train_loader = train_all_epochs(
        model,
        example_end_data["dataframe"].data.values.tolist(),
        example_end_data["dataframe"].data.values.tolist(),
        optimizer,
        example_end_data["scaler_dict"],
        tc_ensemble,
        # break_bad_grads is set to True below because, prior to June 2nd 2022,
        # to it was implicitly set to True.
        break_bad_grads=True,
    )  # warm up SWA
    start_linear_params, start_norm_params = copy_params(model)
    train_end(model, train_loader, optimizer, tc_ensemble)
    end_linear_params, end_norm_params = copy_params(model)
    assert all(
        [
            not torch.equal(start, end)
            for start, end in zip(start_linear_params, end_linear_params)
        ]
    )
    # check that batch norm parameters were updated
    assert all(
        [
            not torch.equal(start, end)
            for start, end in zip(start_norm_params, end_norm_params)
        ]
    )

    hps.set_values(
        {
            "norm": identity,
        }
    )
    model = models.MlpOut(input_dim, 1, hps)
    optimizer = get_optimizer(model, hps)
    # check that swap_swa_sgd was performed
    _, train_loader = train_all_epochs(
        model,
        example_end_data["dataframe"].data.values.tolist(),
        example_end_data["dataframe"].data.values.tolist(),
        optimizer,
        example_end_data["scaler_dict"],
        tc_ensemble,
        # break_bad_grads is set to True below because, prior to June 2nd 2022,
        # to it was implicitly set to True.
        break_bad_grads=True,
    )  # warm up SWA
    start_linear_params, start_norm_params = copy_params(model)
    train_end(model, train_loader, optimizer, tc_ensemble)
    end_linear_params, end_norm_params = copy_params(model)
    assert all(
        [
            not torch.equal(start, end)
            for start, end in zip(start_linear_params, end_linear_params)
        ]
    )
    # check that batch norm parameters were not updated
    assert all(
        [
            torch.equal(start, end)
            for start, end in zip(start_norm_params, end_norm_params)
        ]
    )


def test_compute_batch_size_range():
    min_passes = 10
    max_batch_size = 100
    # test that max_batch is used as upper limit if small enough
    n_data = 1500
    rng = compute_batch_size_range(min_passes, max_batch_size, n_data)
    assert rng == (0.25 * max_batch_size, max_batch_size)
    # test that max_batch is not used as upper limit if it is too big
    n_data = 900
    rng = compute_batch_size_range(min_passes, max_batch_size, n_data)
    assert rng == (
        int(round(0.25 * 900 * (1 / min_passes))),
        int(round(900 * (1 / min_passes))),
    )


def test_UOut():
    x = torch.randn((5,))
    hps = HpConfig()
    beta = ModelParameter(float)
    beta.set_value(0.05)
    setattr(hps, "beta", beta)
    layer = UOut(hps)
    layer.train()
    assert not torch.equal(x, layer(x))
    layer.eval()
    assert torch.equal(x, layer(x))


def test_choose_model_size_by_overfit(example_data):
    hps = HpConfig()
    hps.set_values(
        {
            "r_learn": 0.01,
            "batch_size": 10,
            "dropout_pct": 0.05,
            "capacity": 0,
            "activation": nn.PReLU(),
            "norm": nn.BatchNorm1d,
            "initializer": nn.init.kaiming_normal_,
            "optimizer": "swa",
            "swa_start_frac": 0.2,
            "swa_freq": 5,
            "weight_decay": 0.0,
        }
    )
    tc_overfit = example_data["base_train_config"]
    tc_overfit.hps = hps
    tc_overfit.epochs = 3
    root = "overfit/"
    dataframe, _ = prepare_train(
        example_data["dataframe"], morgan_featurizer, root_dir=root
    )
    dataset = dataframe.data.values.tolist()
    choose_model_size_by_overfit(
        model_class=models.MlpOut,
        model_kwargs_dict={
            "input_dim": st_utils.get_input_dim(dataset[0]),
            "output_dim": 1,
        },
        capacity_ls=list(range(1, 3)),
        data_set=dataset,
        train_config=tc_overfit,
    )
    # if above is run to completion, the test is passed
    assert True


def test_Initializer():
    hps = HpConfig()
    # Check that Initializer will raise an error if an initializer that
    # does not end in "_" is passed in.
    with pytest.raises(AssertionError):
        hps.set_values(
            {
                "initializer": nn.init.kaiming_normal,
            }
        )


def test_initialize_weights():
    """
    Check that the variance of kaiming and xavier are not close to one another.
    """
    diff_threshold = 1.0
    weight = torch.randn(512, 256)
    weight_copy = weight.clone()
    activation = nn.functional.leaky_relu
    gain = nn.init.calculate_gain("leaky_relu", LEAKY_RELU_DEFAULT_SLOPE)
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight)
    xavier_std = gain * np.sqrt(2 / (fan_in + fan_out))
    kaiming_std = gain / np.sqrt(fan_out)
    diff_threshold = 0.8 * np.abs(xavier_std - kaiming_std)
    initialize_weights(weight, nn.init.kaiming_normal_, activation)
    initialize_weights(weight_copy, nn.init.xavier_normal_, activation)
    assert (weight.var().sqrt() - weight_copy.var().sqrt()).abs() > diff_threshold


def test_HpConfig_set_values_from_string():
    hps = HpConfig()
    dictionary = {
        "capacity": 3,
        "dropout_pct": 0.0,
        "activation": nn.functional.leaky_relu,
        "initializer": nn.init.kaiming_normal_,
        "norm": identity,
        "optimizer": "adam",
        "r_learn": 0.0,
        "swa_freq": 0,
        "swa_start_frac": 0.0,
        "batch_size": 0,
        "weight_decay": 0.0,
    }
    hps.set_values(dictionary)
    result = HpConfig()
    extras = {
        "leaky_relu": nn.functional.leaky_relu,
        "kaiming_normal_": nn.init.kaiming_normal_,
        "identity": identity,
    }
    result.set_values_from_string(str(hps), extras)
    assert str(result) == str(hps)


def test_weight_decay(example_data, example_end_data):
    hps = HpConfig()
    hps.set_values(
        {
            "r_learn": 0.01,
            "batch_size": 10,
            "dropout_pct": 0.05,
            "capacity": 0,
            "activation": nn.functional.leaky_relu,
            "norm": identity,
            "initializer": nn.init.kaiming_normal_,
            "optimizer": "adam",
            "swa_start_frac": 0.0,
            "swa_freq": 0,
            "weight_decay": 0.0,
        }
    )
    train_config = example_data["base_train_config"]
    train_config.hps = hps
    train_config.epochs = 1
    dataset = example_end_data["dataframe"].data.values.tolist()
    input_dim = st_utils.get_input_dim(dataset[0])
    modelWd0 = models.MlpOut(
        input_dim=input_dim,
        output_dim=1,
        hps=hps,
    )
    modelWd1 = deepcopy(modelWd0)
    manual_seed(0)
    train_submodel(
        modelWd0,
        dataset,
        dataset[0:2],
        example_end_data["scaler_dict"],
        train_config,
    )
    bigWeightSum = 0
    for name, param in modelWd0.named_parameters():
        if "weight" in name:
            bigWeightSum += param.abs().sum().item()

    manual_seed(0)
    hps.set_values({"weight_decay": 1.0})
    train_submodel(
        modelWd1,
        dataset,
        dataset[0:2],
        example_end_data["scaler_dict"],
        train_config,
    )
    smallWeightSum = 0
    for name, param in modelWd1.named_parameters():
        if "weight" in name:
            smallWeightSum += param.abs().sum().item()
    assert smallWeightSum < bigWeightSum


def test_check_series_values(example_data):
    df = deepcopy(example_data["dataframe"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_series_values(
            df,
            constants._F_GRAPH,
        )
    df.graph_feats = [{"a": 1, "b": 1}] * 13 + [{"a": 1, "b": 2}]
    bad_keys = ["a"]
    with pytest.warns(
        UserWarning,
        match=f"{constants._F_GRAPH} contains constant values in columns: {', '.join(bad_keys)}.",
    ):
        check_series_values(
            df,
            constants._F_GRAPH,
        )
    df.graph_feats = [{"a": 2, "b": 2}] * 13 + [{"a": 1, "b": 1}]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_series_values(
            df,
            constants._F_GRAPH,
        )


def test_weight_classes():
    label_frequency = {0: 2, 1: 4, 2: 8}
    correct = np.array([4.0, 2.0, 1.0])
    result = st_utils.weight_classes(label_frequency)
    assert (correct == result).all()

    label_frequency = {0: 4, 1: 2, 2: 8}
    correct = np.array([2.0, 4.0, 1.0])
    result = st_utils.weight_classes(label_frequency)
    assert (correct == result).all()


@pytest.fixture(scope="session", autouse=True)  # this will tell
# Pytest to run the function below after all tests are completed
def cleanup(request):
    # clean up any files that were created
    def remove_data():
        try:
            os.remove("properties.pkl")
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree("st_regression_ensemble/")
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree("mt_regression_ensemble/")
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree("ensemble_linear/")
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree("train_end/")
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree("overfit/")
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree("singlecore/")
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree("multicore/")
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree("st_classification_ensemble//")
        except FileNotFoundError:
            pass

    # execute remove_dirs at the end of the session
    request.addfinalizer(remove_data)
