from torch import nn, zeros, manual_seed, tensor, LongTensor, atleast_1d

from . import constants
from .utils import _assemble_data, get_unit_sequence
from sol_trainer import layers
from .std_module import StandardModule


class MlpOut(StandardModule):
    """
    This is simply an Mlp layer followed by a my_output layer
    """

    def __init__(self, input_dim, output_dim, hps, debug=False):
        super().__init__(hps)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hps = hps
        self.debug = debug

        self.unit_sequence = get_unit_sequence(
            self.input_dim, self.output_dim, self.hps.capacity.get_value()
        )
        self.mlp = layers.Mlp(
            None,
            None,
            self.hps,
            self.debug,
            self.unit_sequence[:-1],
        )
        self.outlayer = layers.my_output(self.unit_sequence[-2], self.unit_sequence[-1])

    def forward(self, data):
        data.yhat = self.mlp(data.yhat)
        data.yhat = self.outlayer(data.yhat)
        return data


class MlpRegressor(MlpOut):
    def __init__(self, input_dim, output_dim, hps, debug=False):
        super().__init__(input_dim, output_dim, hps, debug)


class MlpClassifier(MlpOut):
    def __init__(self, input_dim, output_dim, hps, debug=False):
        super().__init__(input_dim, output_dim, hps, debug)


class LinearEnsemble(nn.Module):
    """
    An ensemble that takes a straight average of the predictions
    of its submodels.
    """

    def __init__(
        self, submodel_dict, device, scalers, monte_carlo=True, regression=True
    ):
        super().__init__()
        self.submodel_dict = submodel_dict
        self.device = device
        self.scalers = scalers  # dictionary
        self.monte_carlo = monte_carlo
        self.regression = regression

    def forward(self, data, n_passes=None):
        """
        Forward pass of this model.
        """
        # The forward pass of ensembles should always return the prediction
        # *mean* and the prediction *standard deviation*
        manual_seed(constants.RANDOM_SEED)
        # We set the seed above so that all forward passes are reproducible
        batch_size = data.num_graphs
        n_submodels = len(self.submodel_dict)
        if self.monte_carlo:
            all_model_means = (
                zeros((n_submodels, batch_size, self.submodel_dict[0].output_dim))
                .squeeze(-1)
                .to(self.device)
            )
            all_model_vars = (
                zeros((n_submodels, batch_size, self.submodel_dict[0].output_dim))
                .squeeze(-1)
                .to(self.device)
            )
            # TODO: Parallelize?
            for i, model in self.submodel_dict.items():
                model_passes = (
                    zeros((n_passes, batch_size, model.output_dim))
                    .squeeze(-1)
                    .to(self.device)
                )
                for j in range(n_passes):
                    data = model(_assemble_data(model, data))
                    data.yhat = data.yhat
                    if self.regression:
                        data.yhat = tensor(
                            [
                                self.scalers[str(ind_prop)].inverse_transform(val)
                                for (ind_prop, val) in zip(data.prop, data.yhat)
                            ]
                        )
                    model_passes[j] = data.yhat

                all_model_means[i] = model_passes.mean(dim=0)
                all_model_vars[i] = model_passes.var(dim=0)

            data.yhat = all_model_means.mean(dim=0).squeeze()
            data.yhat_std = (
                all_model_vars + all_model_means.square() - data.yhat.square()
            ).mean(dim=0)
            data.yhat_std = data.yhat_std.sqrt().squeeze()
        else:
            all_model_preds = (
                zeros((n_submodels, batch_size, self.submodel_dict[0].output_dim))
                .squeeze(-1)
                .to(self.device)
            )
            # TODO: Parallelize?
            for i, model in self.submodel_dict.items():
                data = model(_assemble_data(model, data))
                if self.regression:
                    data.yhat = tensor(
                        [
                            self.scalers[str(ind_prop)].inverse_transform(val)
                            for (ind_prop, val) in zip(data.prop, data.yhat)
                        ]
                    )

                all_model_preds[i] = data.yhat
            data.yhat = all_model_preds.mean(dim=0).squeeze()
            data.yhat_std = all_model_preds.std(dim=0).squeeze()
        if not self.regression:
            # For classification, data.y_hat contains raw class scores
            # at this point. We need to convert these scores into a
            # class label. We do so by taking the argmax.
            data.yhat = atleast_1d(data.yhat.max(-1)[1])
        return data
