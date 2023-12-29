from torch import tensor, equal
from torch import save as torch_save
from torch import float as torch_float
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
import numpy as np
from copy import deepcopy
from sol_trainer.std_module import StandardModule
from sol_trainer.infer import eval_ensemble

n_features = 512  # hard-coded


class MathModel1(StandardModule):
    def __init__(self, hps, dummy_attr):
        super().__init__(hps)
        self.hps = hps
        self.dummy_attr = dummy_attr
        self.output_dim = 1

    def forward(self, data):
        data.yhat = (
            data.yhat[:, 0] + data.yhat[:, 1] + data.yhat[:, 2] - data.yhat[:, 3]
        )
        return data


class MathModel2(StandardModule):
    def __init__(self, hps):
        super().__init__(hps)
        self.hps = hps
        self.output_dim = 1

    def forward(self, data):
        data.yhat = data.yhat[:, 0] - data.yhat[:, 1]

        return data


class MathModel3(StandardModule):
    def __init__(self, hps):
        super().__init__(hps)
        self.hps = hps
        self.output_dim = 1

    def forward(self, data):
        data.yhat = data.yhat[:, 0] + data.yhat[:, 1]

        return data


def trainer_MathModel(model, train_pts, val_pts, scaler_dict, tc, break_bad_grads=True):
    """
    break_bad_grads is set to True here because, prior to June 2nd 2022, to it was implicitly set to True.
    """
    # this function will "train" the model and save it
    if tc.model_save_path:
        torch_save(model.state_dict(), tc.model_save_path)
        print("Best model saved", flush=True)


def morgan_featurizer(smile):
    smile = smile.replace("*", "H")
    mol = Chem.MolFromSmiles(smile)
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=n_features, useChirality=True
    )
    fp = np.expand_dims(fp, 0)
    return Data(x=tensor(fp, dtype=torch_float))


def has_bn(model):
    has_bn = False
    for m in model.modules():
        if m.__class__.__name__.startswith("BatchNorm1d"):
            has_bn = True
            break
    return has_bn


def copy_params(model):
    linear_params = deepcopy(
        [x[1] for x in list(model.named_parameters()) if "linear.weight" in x[0]]
    )
    norm_params = deepcopy(
        [x[1] for x in list(model.named_parameters()) if "norm.weight" in x[0]]
    )

    return linear_params, norm_params


def get_prelu_slopes(model):
    return deepcopy(
        [
            x[1].item()
            for x in list(model.named_parameters())
            if "activation.weight" in x[0]
        ]
    )


def scaler_util(scaler, data, true):
    """
    A utility function that performs two tests.
    1) That scaler.fit_transform(data) is equal to true.
    2) That scaler.inverse_transform(true) is equal to data.
    """
    ### Test the transform.
    result = scaler.fit_transform(data)
    assert equal(result, true)
    ### Test the inverse transform
    assert equal(scaler.inverse_transform(true), data)


def scaler_fromstring_util(scaler):
    """
    This function checks that a new Scaler object created from a
    string of an old Scaler object is equal to that old Scaler object.
    """
    scaler_string = str(scaler)
    new_scaler = type(scaler).from_string(scaler_string)
    assert scaler_string == str(new_scaler)


def ensemble_trainer_helper(ensemble, root, data0, data1):
    _, mc_result0, _, _ = eval_ensemble(
        ensemble,
        root,
        data0,
        morgan_featurizer,
        "cpu",
        ensemble_forward_kwargs={"n_passes": 2},
    )
    _, mc_result1, _, _ = eval_ensemble(
        ensemble,
        root,
        data1,
        morgan_featurizer,
        "cpu",
        ensemble_forward_kwargs={"n_passes": 2},
    )
    return mc_result0, mc_result1
