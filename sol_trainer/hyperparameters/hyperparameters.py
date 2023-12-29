import numpy as np
from torch_geometric.loader import DataLoader
from torch import optim
from torchcontrib.optim import SWA
from torch import nn
from sol_trainer import loss as loss_module
from sol_trainer import constants, scale, utils

passable_types = {
    int: [int, np.int64],
    float: [float],
    scale.SequentialScaler: [scale.SequentialScaler],
}


class identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class Parameter:
    def __init__(self, pass_type, options=None):
        self.pass_type = pass_type
        self.value = None
        self.options = options  # acceptable string values

    def set_value(self, value):
        if self.pass_type == str:
            if value in self.options:
                self.value = value
            else:
                raise TypeError(
                    f"The value passed in is '{value}' but the only valid options are {self.options}."
                )
        elif self.pass_type == scale.SequentialScaler:
            self.value = value
        elif self.pass_type == callable:
            if callable(value):
                self.value = value
            else:
                raise TypeError(f"The value passed in is not callable.")
        elif any(
            [
                isinstance(value, valid_type)
                for valid_type in passable_types[self.pass_type]
            ]
        ):
            self.value = self.pass_type(value)
        else:
            raise TypeError(
                f"The value passed in is of type {type(value)} but it should be of type {self.pass_type}"
            )
        self.set_value_callbacks()

    def get_value(self):
        return self.value

    def set_value_callbacks(self):
        """
        This function will be run after set_value
        """
        pass

    def __str__(self):
        if self.pass_type == callable:
            if self.value:
                if hasattr(self.value, "__name__"):
                    return self.value.__name__
                else:
                    # PReLU object, for example, will not have __name__. So
                    # let's just hope PyTorch gives a good string
                    # representation of the object
                    return str(self.value)
            else:
                return str(None)
        else:
            return str(self.value)


class ModelParameter(Parameter):
    def __init__(self, pass_type):
        super().__init__(pass_type)


class HpConfig:
    """
    A class to configure which hyperparameters to use during training
    """

    def __init__(self):
        super().__init__()
        # below are "standard" hyoerparameters that should always be set
        self.capacity = ModelParameter(int)
        self.batch_size = Parameter(int)
        self.r_learn = Parameter(float)
        self.dropout_pct = ModelParameter(float)  # between 0 and 1
        # initialize an activation
        self.activation = ModelParameter(callable)
        self.activation.set_value(nn.functional.leaky_relu)
        # initialize a normalization method
        self.norm = ModelParameter(callable)
        self.norm.set_value(identity)
        # initialize an initialization method
        self.initializer = Initializer()
        self.initializer.set_value(nn.init.xavier_uniform_)
        # initialize the optimizer
        self.optimizer = Parameter(str, ["adam", "swa"])
        self.optimizer.set_value("adam")
        # initialize swa parameters
        self.swa_start_frac = Parameter(float)  # between 0 and 1
        self.swa_start_frac.set_value(0.0)
        self.swa_freq = Parameter(int)
        self.swa_freq.set_value(0)
        # initialize a weight decay value
        self.weight_decay = Parameter(float)
        self.weight_decay.set_value(0.0)
        # initialize a scaler for graph features
        scaler = Parameter(scale.SequentialScaler)
        scaler.set_value(scale.SequentialScaler())
        graph_scaler_name = f"{constants._F_GRAPH}_scaler"
        setattr(self, graph_scaler_name, scaler)
        # initialize a scaler for node features
        scaler = Parameter(scale.SequentialScaler)
        scaler.set_value(scale.SequentialScaler())
        node_scaler_name = f"{constants._F_NODE}_scaler"
        setattr(self, node_scaler_name, scaler)

    def set_values(self, dictionary):
        for key, val in dictionary.items():
            if hasattr(self, key):
                param = getattr(self, key)
                param.set_value(val)

    def __str__(self):
        attrs = utils.sorted_attrs(self)
        attrs_str = "; ".join([f"{k}: {v}" for k, v in attrs])
        return "{" + attrs_str + "}"

    def set_values_from_string(
        self,
        string,
        extras={
            "leaky_relu": nn.functional.leaky_relu,
            "kaiming_normal_": nn.init.kaiming_normal_,
            "kaiming_uniform_": nn.init.kaiming_uniform_,
            "identity": identity,
            "xavier_uniform_": nn.init.xavier_uniform_,
            "xavier_normal_": nn.init.xavier_normal_,
        },
    ):
        string = string.replace("{", "").replace("}", "")
        attrs_list = string.split("; ")
        dictionary = {}
        for attr in attrs_list:
            name, value = tuple(attr.split(": ", 1))
            pass_type = getattr(self, name).pass_type
            if pass_type == scale.SequentialScaler:
                scaler = scale.SequentialScaler()
                scaler.from_string(value)
                value = scaler
            elif pass_type == callable:
                value = extras[value]
            else:
                if value == "None":
                    value = None
                else:
                    value = pass_type(value)
            dictionary[name] = value

        self.set_values(dictionary)


# #############################
# Code related to batch size
# #############################
def _compute_batch_max(model, data_ls, device, curr_size, forward_kwargs):
    """
    Helper function for compute_batch_max
    """
    loader = DataLoader(data_ls, batch_size=curr_size)
    # We need to determine if model will require backpropogation. If not,
    # then we can potentially send more data to the GPU. To check, we will
    # see if the model has any parameters and if they require gradients.
    param_list = list(model.parameters())
    if len(param_list) == 0:
        # No parameters in model, so no need to backprop.
        backpropagate = False
    else:
        if any([x.requires_grad for x in param_list]):
            # There are parameters, and at least one requires a gradient.
            # Thus, we need to backprop.
            backpropagate = True
        else:
            # There are parameters, but none require gradients. We do not
            # need backprop.
            backpropagate = False
    for data in loader:
        data.to(device)
        if backpropagate:
            optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimization
            loss_obj = loss_module.sh_mse_loss()
        output = model(utils._assemble_data(model, data), **forward_kwargs)
        if backpropagate:
            loss = loss_obj(output, data)
            loss.backward()
            optimizer.step()
        # Break below because we only need to test one batch. Testing more
        # batches will only take more time.
        break


def compute_batch_max(model, data_ls, device, forward_kwargs={}, init_size=4096):
    frac = 0.5  # the fraction of curr_size to return, in order to
    # handle slightly bigger or slightly smaller data_ls seen during
    # training
    model.to(device)
    curr_size = init_size
    while curr_size > 1:
        try:
            _compute_batch_max(model, data_ls, device, curr_size, forward_kwargs)
            break
        except RuntimeError as e:
            e_str = str(e)
            if "CUDA out of memory" in e_str:
                e_str = "CUDA out of memory error"
            else:
                raise e
            print(f"Received {e_str} with batch size equal to {curr_size}.")
            curr_size = curr_size // 2
    return int(np.floor(frac * curr_size)) + 1


def compute_batch_size_range(min_passes, max_batch_size, n_data):
    upper_limit = int(round(min((1 / min_passes) * n_data, max_batch_size)))
    lower_limit = int(round(0.25 * upper_limit))
    # Make sure batch size lower bound is at least 2 for batch norm.
    lower_limit = max(lower_limit, 2)
    # Make sure batch size upper bound is greater than the lower bound.
    if upper_limit <= lower_limit:
        upper_limit = lower_limit + 1

    return (lower_limit, upper_limit)


# ###########################
# Code related to optimizers
# ###########################
def get_optimizer(model, hps: HpConfig):
    string = hps.optimizer.get_value()
    params = add_weight_decay(model, hps.weight_decay.get_value())
    if string == "adam":
        return optim.Adam(
            params,
            lr=hps.r_learn.get_value(),
        )
    elif string == "swa":
        return SWA(
            optim.SGD(
                params,
                lr=hps.r_learn.get_value(),
            )
        )


def add_weight_decay(model, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or ".bias" in name or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": l2_value},
    ]


def is_SWA(optimizer):
    if isinstance(optimizer, SWA):
        return True
    else:
        return False


def update_optimizer(optimizer, hps, total_epochs, curr_epoch):
    if is_SWA(optimizer):
        progress = curr_epoch / total_epochs
        if (progress > hps.swa_start_frac.get_value()) and (
            curr_epoch % hps.swa_freq.get_value() == 0
        ):
            optimizer.update_swa()


# #########################################
# Code related to parameter initialization
# #########################################
class Initializer(ModelParameter):
    def __init__(self):
        super().__init__(callable)

    def set_value_callbacks(self):
        # In PyTorch, functions that end in "_" denote operations that
        # modify tensors in place. For initializers, we only support these
        # type of in-place operations.
        assert self.value.__name__.endswith("_")
