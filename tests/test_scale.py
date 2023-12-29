from shutil import rmtree
import pytest
from sol_trainer.hyperparameters.hyperparameters import HpConfig
from sol_trainer.infer import eval_ensemble
from sol_trainer.load import load_ensemble
from sol_trainer.models import LinearEnsemble
from sol_trainer.prepare import prepare_train
from sol_trainer.scale import *
import numpy as np
from torch import tensor, equal
from torch import float as torch_float
from sol_trainer.train import train_kfold_ensemble, trainConfig
from .utils_test import MathModel3, scaler_fromstring_util, scaler_util
import pandas as pd
from torch_geometric.loader import DataLoader

root_dirs = []


def test_basic():
    data = np.array([[100], [0.01], [1]])
    MyScaler = SequentialScaler()
    MyScaler.append(LogTenScaler())
    MyScaler.append(MinMaxScaler())
    data_trans = MyScaler.fit_transform(data)
    out = MyScaler.inverse_transform(data_trans)
    assert (out == tensor(data, dtype=torch_float)).all()


def test_transform():

    data = np.array([[100], [0.01], [1]])
    MyScaler = SequentialScaler()
    MyScaler.append(LogTenScaler())
    MyScaler.append(MinMaxScaler())
    data_fittrans = MyScaler.fit_transform(data)
    data_trans = MyScaler.transform(data)
    assert (data_fittrans == data_trans).all()


def test_ZeroMeanScaler():

    data = np.array([[100], [0.01], [1]])
    MyScaler = SequentialScaler()
    MyScaler.append(ZeroMeanScaler())
    MyScaler.append(MinMaxScaler())
    data_fittrans = MyScaler.fit_transform(data)
    data_trans = MyScaler.transform(data)
    assert (data_fittrans == data_trans).all()

    # Test the case that data is 2D and we want to center the data
    # on the mean of each feature.
    data = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    MyScaler = SequentialScaler()
    MyScaler.append(ZeroMeanScaler(dim=-2))
    ### Check that inverse_transform cannot be called before fit_transform.
    with pytest.raises(AttributeError):
        MyScaler.inverse_transform(data)
    ### Test the transform and inverse_transform.
    true = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    scaler_util(MyScaler, data, true)

    # Test the case that data is 2D and we want to center the data
    # on the mean of each row.
    data = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    MyScaler = SequentialScaler()
    MyScaler.append(ZeroMeanScaler(dim=-1))
    ### Test the transform and inverse_transform.
    true = torch.tensor([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    scaler_util(MyScaler, data, true)

    # Test the from_string method.
    scaler_fromstring_util(MyScaler.scaler_ls[0])


def test_MinMaxScaler():
    # Test the case that data is 2D and we want to MinMax scale each
    # column. We will test both the transform and inverse_transform.
    data = torch.tensor([[1.0, 2.0], [0.0, 0.0], [-1.0, 2.0]])
    MyScaler = SequentialScaler()
    MyScaler.append(MinMaxScaler(dim=-2))
    true = torch.tensor([[1.0, 1.0], [0.5, 0.0], [0.0, 1.0]])
    scaler_util(MyScaler, data, true)

    # Test the case that data is 2D and we want to MinMax scale each
    # row. We will test both the transform and inverse_transform.
    data = torch.tensor([[-1.0, 0.0, 1.0], [0.0, 4.0, 10.0]])
    MyScaler = SequentialScaler()
    MyScaler.append(MinMaxScaler(dim=-1))
    true = torch.tensor([[0.0, 0.5, 1.0], [0.0, 0.4, 1.0]])
    scaler_util(MyScaler, data, true)


def test_QuotientScaler():

    # Test 1D data.
    data = tensor([[10], [8]], dtype=torch_float)
    MyScaler = SequentialScaler()
    MyScaler.append(QuotientScaler(2))
    data_scale = MyScaler.fit_transform(data)
    assert (data_scale == tensor([[5], [4]], dtype=torch_float)).all()

    data_unscale = MyScaler.inverse_transform(data_scale)

    assert (data_unscale == data).all()

    # Test the case that data is 2D and we want to divide each feature
    # by a different number. We will test both the transform and
    # inverse_transform.
    data = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    MyScaler = SequentialScaler()
    MyScaler.append(QuotientScaler(torch.tensor([1.0, 2.0, 3.0]), dim=-2))
    true = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    scaler_util(MyScaler, data, true)

    # Test the case that data is 2D and we want to divide each row
    # by a different number. We will test both the transform and
    # inverse_transform.
    data = torch.tensor([[4.0, 8.0, 12.0], [4.0, 8.0, 12.0]])
    MyScaler = SequentialScaler()
    MyScaler.append(QuotientScaler(torch.tensor([2.0, 4.0]), dim=-1))
    true = torch.tensor([[2.0, 4.0, 6.0], [1.0, 2.0, 3.0]])
    scaler_util(MyScaler, data, true)

    # Test the from_string method.
    scaler_fromstring_util(MyScaler.scaler_ls[0])


def test_ProductScaler():

    data = tensor([[10], [8]], dtype=torch_float)
    MyScaler = SequentialScaler()
    MyScaler.append(ProductScaler(2))
    data_scale = MyScaler.fit_transform(data)
    assert (data_scale == tensor([[20], [16]], dtype=torch_float)).all()

    data_unscale = MyScaler.inverse_transform(data_scale)

    assert (data_unscale == data).all()


def test_islinear():

    MyScaler = SequentialScaler()
    MyScaler.append(LogTenScaler())
    MyScaler.append(MinMaxScaler())
    assert MyScaler.is_linear() == False

    MyScaler = SequentialScaler()
    MyScaler.append(MinMaxScaler())
    assert MyScaler.is_linear() == True


def test_string(capsys):

    data = np.array([[100], [0.01], [1]])
    MyScaler = SequentialScaler()
    MyScaler.append(LogTenScaler())
    MyScaler.append(MinMaxScaler())
    MyScaler.fit_transform(data)
    correct_string = "Forward(LogTenScaler(dim: 0) --> MinMaxScaler(dim: 0, max: tensor([2.]), min: tensor([-2.])))"
    assert str(MyScaler) == correct_string
    print(MyScaler)
    captured = capsys.readouterr()
    assert captured.out.strip() == correct_string


def test_SequentialScaler_from_string():
    """
    Create a sequential scaler from a string.
    """
    start = SequentialScaler()
    start.append(LogTenScaler())
    mm = MinMaxScaler()
    mm.fit([1, 2, 3])
    start.append(mm)
    result = SequentialScaler()
    result.from_string(str(start))
    assert str(result) == str(start)

    data = [1000, 100, 10]

    assert equal(start.transform(data), result.transform(data))


def test_feature_scaling():
    """
    Test that feature scaling will work as we expect when training
    ensemble models.
    """
    graph_feats = [
        {"feat0": -1, "feat1": 2},
        {"feat0": 0, "feat1": 2},
        {"feat0": 1, "feat1": 0},
    ]
    props = ["prop1"] * len(graph_feats)
    values = [i for i in range(len(graph_feats))]
    data = {"prop": props, "value": values, "graph_feats": graph_feats}
    graph_feats_scaler = SequentialScaler()
    graph_feats_scaler.append(MinMaxScaler(dim=-2))
    hps = HpConfig()
    hps.set_values(
        {
            "graph_feats_scaler": graph_feats_scaler,
            "capacity": -1,  # dummy value
            "batch_size": -1,  # dummy value
            "r_learn": 0.0,  # dummy value
            "dropout_pct": 0.0,  # dummy value
        }
    )
    root_dir = "ensemble_scaled_features"
    root_dirs.append(root_dir)
    dataframe = pd.DataFrame(data)
    dataframe_processed, scaler_dict = prepare_train(
        dataframe,
        smiles_featurizer=None,
        root_dir=root_dir,
        hps=hps,
    )
    tc = trainConfig(
        loss_obj=None,
        amp=False,
    )
    tc.device = "cpu"  # run this on a CPU since this is just a test
    tc.hps = hps

    def trainer(model, train_pts, val_pts, scaler_dict, tc, break_bad_grads=False):
        # This function will "train" the model and save it.
        if tc.model_save_path:
            torch.save(model.state_dict(), tc.model_save_path)
            print("Best model saved", flush=True)
        train_loader = DataLoader(train_pts, batch_size=len(dataframe))
        for data in train_loader:
            if tc.fold_index == 0:
                assert torch.equal(data.graph_feats, torch.tensor([[0.0, 1.0]]))
            elif tc.fold_index == 1:
                assert torch.equal(
                    data.graph_feats, torch.tensor([[0.5, 1.0], [1.0, 0.0]])
                )
            break

    train_kfold_ensemble(
        dataframe_processed,
        submodel_cls=MathModel3,
        submodel_kwargs_dict={"hps": hps},
        train_config=tc,
        submodel_trainer=trainer,
        augmented_featurizer=None,  # since we do not want augmentation
        scaler_dict=scaler_dict,
        root_dir=root_dir,
        n_fold=2,
        random_seed=234,
    )
    # check that tc.get_train_pts is set to None
    assert tc.get_train_pts == None
    # ####################################
    # load the ensemble and run it forward
    # ####################################
    ensemble = load_ensemble(
        LinearEnsemble,
        root_dir,
        submodel_cls=MathModel3,
        device="cpu",
        submodel_kwargs_dict={},
    )
    _, mean, _, _ = eval_ensemble(
        model=ensemble,
        root_dir=root_dir,
        dataframe=dataframe,
        smiles_featurizer=None,
        device="cpu",
        ensemble_kwargs_dict={"n_passes": 1},
    )
    np.testing.assert_allclose(mean, np.array([1, 1.5, 1]), rtol=1e-5, atol=0)


@pytest.fixture(scope="session", autouse=True)  # this will tell
# Pytest to run the function below after all tests are completed
def cleanup(request):
    # clean up any files that were created
    def remove_data():
        for dir in root_dirs:
            try:
                rmtree(dir)
            except FileNotFoundError:
                pass

    # execute remove_dirs at the end of the session
    request.addfinalizer(remove_data)
