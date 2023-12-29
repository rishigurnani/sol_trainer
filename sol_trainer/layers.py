from torch import nn
import inspect
from sol_trainer import std_module
from sol_trainer import utils

LEAKY_RELU_DEFAULT_SLOPE = 0.01


def calculate_gain(activation):
    if activation == nn.functional.leaky_relu:
        params = (
            "leaky_relu",
            LEAKY_RELU_DEFAULT_SLOPE,
        )  # negative slope is hard-coded at 0.01
    elif activation == nn.functional.tanh:
        params = "tanh"
    elif activation == nn.functional.relu:
        params = "relu"
    elif activation == nn.functional.selu:
        params = "selu"
    elif activation == nn.functional.sigmoid:
        params = "sigmoid"
    else:
        raise ValueError("Unsupported nonlinearity {}".format(activation))
    return nn.init.calculate_gain(*params)


def initialize_bias(bias_data, activation):
    if isinstance(activation, nn.PReLU):
        pass
    else:
        if activation == nn.functional.relu:
            a = 0.1  # use a value > 0 to prevent saturation of the ReLU at initialization
        else:
            a = 0.0
        nn.init.constant_(bias_data, a)


def initialize_weights(weight, initializer, activation):
    if "gain" in inspect.getfullargspec(initializer)[0]:
        initializer(weight, gain=calculate_gain(activation))
    else:
        if initializer == nn.init.kaiming_normal_:
            if activation == nn.functional.leaky_relu:
                a = LEAKY_RELU_DEFAULT_SLOPE
            elif isinstance(activation, nn.PReLU):
                a = activation.weight.item()
            else:
                raise (f"Invalid activation {activation} passed in for {initializer}.")
            initializer(weight, a=a, mode="fan_out", nonlinearity="leaky_relu")
        else:
            raise (f"Invalid initializer {initializer} passed in.")


class identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class my_hidden(std_module.StandardModule):
    """
    Hidden layer with xavier initialization and batch norm
    """

    def __init__(self, size_in, size_out, hps):
        super().__init__(hps)
        self.size_in, self.size_out = size_in, size_out
        self.linear = nn.Linear(self.size_in, self.size_out)
        self.activation = hps.activation.get_value()
        self.bn = nn.BatchNorm1d(self.size_out)
        self.initializer = hps.initializer.get_value()
        initialize_weights(self.linear.weight, self.initializer, self.activation)
        initialize_bias(self.linear.bias.data, self.activation)

    def forward(self, x):
        if self.activation is not None:
            return self.bn(self.activation(self.linear(x)))
        else:
            return self.bn(self.linear(x))


class my_hidden2(std_module.StandardModule):
    """
    Hidden layer with xavier initialization and dropout
    """

    def __init__(self, size_in, size_out, hps):
        super().__init__(hps)
        self.size_in, self.size_out = size_in, size_out
        self.linear = nn.Linear(self.size_in, self.size_out)
        self.activation = hps.activation.get_value()
        self.dropout = nn.Dropout(hps.dropout_pct.get_value())
        self.initializer = hps.initializer.get_value()
        initialize_weights(self.linear.weight, self.initializer, self.activation)
        initialize_bias(self.linear.bias.data, self.activation)

    def forward(self, x):
        if self.activation is not None:
            return self.dropout(self.activation(self.linear(x)))
        else:
            return self.dropout(self.linear(x))


class my_hidden3(std_module.StandardModule):
    """
    Hidden layer with xavier initialization, dropout, and normalization
    """

    def __init__(self, size_in, size_out, hps):
        super().__init__(hps)
        self.size_in, self.size_out = size_in, size_out
        self.linear = nn.Linear(self.size_in, self.size_out)
        self.activation = hps.activation.get_value()
        self.norm = hps.norm.get_value()(self.size_out)
        self.initializer = hps.initializer.get_value()
        initialize_weights(self.linear.weight, self.initializer, self.activation)
        initialize_bias(self.linear.bias.data, self.activation)
        # if we are normalizing, then we need to avoid using a high
        # dropout percent that will drastically change the variance
        # between train and test time
        if not isinstance(self.norm, identity):
            dropout_pct = min(hps.dropout_pct.get_value(), 0.05)
        else:
            dropout_pct = hps.dropout_pct.get_value()
        self.dropout = nn.Dropout(dropout_pct)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class my_output(std_module.StandardModule):
    """
    Output layer with xavier initialization on weights
    Output layer with target mean (plus noise) on bias. Suggestion from: http://karpathy.github.io/2019/04/25/recipe/
    """

    def __init__(self, size_in, size_out, target_mean=None):
        super().__init__(None)
        self.size_in, self.size_out = size_in, size_out
        self.target_mean = target_mean

        self.linear = nn.Linear(self.size_in, self.size_out)
        nn.init.xavier_uniform_(self.linear.weight, gain=1.0)
        if self.target_mean != None:
            self.linear.bias.data.uniform_(0.99 * target_mean, 1.01 * target_mean)
        else:
            nn.init.constant_(self.linear.bias.data, 0)

    def forward(self, x):
        return self.linear(x)


class Mlp(std_module.StandardModule):
    """
    A Feed-Forward neural Network that uses DenseHidden layers
    """

    def __init__(self, input_dim, output_dim, hps, debug, unit_sequence=None):
        super().__init__(hps)
        self.debug = debug
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layers = nn.ModuleList()
        if unit_sequence:
            self.unit_sequence = unit_sequence
            # if the unit_sequence is passed in then several other
            # attributes should be reset to be compatible with the
            # data in unit_sequence
            self.input_dim = self.unit_sequence[0]
            self.output_dim = self.unit_sequence[-1]
            self.hps.capacity.set_value(len(self.unit_sequence) - 2)
        else:
            self.unit_sequence = utils.get_unit_sequence(
                input_dim, output_dim, self.hps.capacity.get_value()
            )
        # set up hidden layers
        for ind, n_units in enumerate(self.unit_sequence[:-1]):
            size_out_ = self.unit_sequence[ind + 1]
            self.layers.append(
                my_hidden3(
                    size_in=n_units,
                    size_out=size_out_,
                    hps=self.hps,
                )
            )

    def forward(self, x):
        """
        Compute the forward pass of this model
        """
        for layer in self.layers:
            x = layer(x)

        return x


class UOut(std_module.StandardModule):
    """
    "Uniform Dropout" - taken from https://arxiv.org/abs/1801.05134
    """

    def __init__(self, hps):
        super().__init__(hps)

    def forward(self, x):
        """
        Compute the forward pass of this block
        """
        if self.training:
            noise = (
                x.clone()
                .detach()
                .uniform_(-self.hps.beta.get_value(), -self.hps.beta.get_value())
            )

            return x + noise

        return x
