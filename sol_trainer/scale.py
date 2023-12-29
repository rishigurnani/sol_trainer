# These scalers are all implemented with PyTorch so that they can be used
# on a GPU, if necessary. These scalers are only written for use
# on ***labels***, not features
import ast
import torch
from .utils import sorted_attrs


ignore_attrs = ["index", "is_parent", "linear"]


class SequentialScaler:
    def __init__(self):
        self.scaler_ls = []  # the list of child scalers
        self.n_children = 0  # how many child scalers are there?
        self.index = 0  # the index of scaler in its parent's sequence
        self.is_parent = True

    def fit(self, data):
        # This method should not be used. Try fit_transform instead.
        raise NotImplementedError()

    def fit_transform(self, data):
        # fit child scalers
        data = self.format_tensorlike(data)
        sequence = sorted(self.scaler_ls, key=lambda x: x.index)
        for scaler in sequence:
            data = scaler.fit_transform(data)

        return data

    def transform(self, data):
        """
        Keyword Arguments:
            data: The data to transform
        """
        data = self.format_tensorlike(data)
        sequence = sorted(self.scaler_ls, key=lambda x: x.index)
        for scaler in sequence:
            data = scaler.transform(data)

        return data

    def inverse_transform(self, data):
        """
        Keyword Arguments:
            data: The data to transform
        """
        data = self.format_tensorlike(data)
        sequence = self.scaler_ls[::-1]  # reverse, reverse!
        for scaler in sequence:
            data = scaler.inverse_transform(data)

        return data

    def append(self, scaler):
        assert isinstance(scaler, Scaler)
        self.scaler_ls.append(scaler)
        setattr(scaler, "index", self.n_children + 1)
        self.n_children += 1
        setattr(scaler, "is_parent", False)

    def is_linear(self):

        return all([scaler.is_linear() for scaler in self.scaler_ls])

    def __str__(self) -> str:
        string = (
            "Forward(" + " --> ".join(str(scaler) for scaler in self.scaler_ls) + ")"
        )

        return string

    def format_tensorlike(self, data):
        """
        Format 'data' as a torch tensor
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float)
        # do not store gradients because we do not need to compute dLoss/dy
        if data.is_leaf:
            data.requires_grad = False
        return data

    def from_string(self, string):
        # Strings prior to June 7th, 2022 contained a colon. The colon
        # was removed after this date.
        if "Forward: " in string:
            string = string.replace("Forward: ", "")
        else:
            if "Forward(" in string:
                string = string.replace("Forward(", "")
                string = string[:-1]  # remove the trailing ")"
        if string:
            str_list = string.split(" --> ")
            for individual_str in str_list:
                if "(" in individual_str:
                    left_paren_idx = individual_str.index("(")
                    scaler_name = individual_str[:left_paren_idx]
                    scaler_cls = globals()[scaler_name]
                    scaler = scaler_cls.from_string(individual_str)
                    self.append(scaler)
                else:
                    # If individual_str does not contain "(" then the
                    # corresponding scaler does not need any inputs to
                    # __init__.
                    scaler = globals()[individual_str]()
                    self.append(scaler)


class SafeSequentialScaler(SequentialScaler):
    """
    A SequentialScaler that will not perform log scaling. This function is
    needed so that high-variance predictions of log properties can be
    computed without being sent to infinity.
    """

    def __init__(self, seq_scaler):
        super().__init__()
        self.unsafe_scaler = seq_scaler
        log_idx = None
        for idx, scaler in enumerate(self.unsafe_scaler.scaler_ls):
            if "Log" in scaler.__class__.__name__:
                log_idx = idx
            else:
                self.append(scaler)
        if log_idx:
            raise ValueError(
                f"{seq_scaler} has a log-based scaler in an unsupported position: {log_idx}"
            )

    def inverse_transform(self, data):
        return super().inverse_transform(data)

    def transform(self, data):
        return super().transform(data)

    def fit(self, data):
        return NotImplementedError()

    def fit_transform(self, data):
        return NotImplementedError()


class Scaler:
    def __init__(self, dim=0):
        """
        Keyword Arguments:
            dim (int): The dimension along which data should be transformed.
        """
        self.linear = False  # should be True if the scaler is a linear
        # transformation (e.g., MinMax) and should if False if the
        # scaler is not a linear transformation (e.g., LogTen). As a
        # default, we will set this value to False.
        self.dim = dim

    def fit(self, data):
        data = self.format_tensorlike(data)
        return data

    def fit_transform(self, data):
        data = self.format_tensorlike(data)
        self.fit(data)
        self.attrs_to_device(data)
        return self.transform(data)

    def transform(self, data):
        """
        Keyword Arguments:
            data: The data to transform
        """
        data = self.format_tensorlike(data)
        self.attrs_to_device(data.device)
        return data

    def inverse_transform(self, data):
        """
        Keyword Arguments:
            data: The data to transform
        """
        data = self.format_tensorlike(data)
        self.attrs_to_device(data.device)
        return data

    @classmethod
    def format_tensorlike(cls, data):
        """
        Format 'data' as a torch tensor.
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float)
        # do not store gradients because we do not need to compute dLoss/dy
        if data.is_leaf:
            data.requires_grad = False
        return data

    @classmethod
    def string_to_tensor(cls, string):
        """
        Format a tensor string to a tensor.
            Ex. string='tensor([1., 2.])'
        """
        return cls.format_tensorlike(
            ast.literal_eval(string.replace("tensor(", "").replace(")", ""))
        )

    def is_linear(self):
        return self.linear

    def __str__(self) -> str:
        attrs = sorted_attrs(self)
        attrs = [x for x in attrs if x[0] not in ignore_attrs]
        self_class_name = type(self).__name__
        if not attrs:
            return self_class_name
        else:
            attr_str = ", ".join([f"{k}: {v}" for k, v in attrs])
            return f"{self_class_name}({attr_str})"

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def dim_from_string(cls, string):
        # get the "dim" value
        if "dim: " in string:
            # extract as dim as an integer-like string
            value = string.split("dim: ")[1].split(",")[0].replace(")", "")
            value = int(value)
        else:
            value = 0
        return value

    @classmethod
    def from_string(cls, string):
        """
        The default from_string method for Scaler objects. This method
        should be overwritten in child classes that contain
        arguments other than "dim" in their __init__ method.
        """
        return cls(dim=cls.dim_from_string(string))

    def attr_to_device(self, attr_name, device):
        setattr(
            self,
            attr_name,
            getattr(self, attr_name).to(device),
        )

    @classmethod
    def tensorlike_attrs(cls):
        """
        A list of attributes of self that should be tensors. By default,
        this list is empty.
        """
        return []

    def attrs_to_device(self, device):
        for attr in self.tensorlike_attrs():
            self.attr_to_device(attr, device)


class ZeroMeanScaler(Scaler):
    def __init__(self, dim=0):
        """
        Keyword Arguments:
            dim (int): The dimension along which data should be transformed.
        """
        super().__init__(dim)
        self.mean = None
        self.linear = True

    @classmethod
    def tensorlike_attrs(cls):
        """
        A list of attributes of self that should be tensors.
        """
        return ["mean"]

    def fit(self, data):
        data = super().fit(data)
        self.mean = torch.clone(data).mean(self.dim)

    def transform(self, data):
        data = super().transform(data)
        return data - self.mean.unsqueeze(self.dim)

    def inverse_transform(self, data):
        """
        Keyword Arguments:
            data: The data to transform
        """
        data = super().inverse_transform(data)
        if isinstance(self.mean, torch.Tensor):
            return data + self.mean.unsqueeze(self.dim)
        else:
            raise AttributeError(
                ".fit_transform() should be called before .inverse_transform()"
            )

    @classmethod
    def from_string(cls, string):
        scaler = cls(dim=cls.dim_from_string(string))
        # set "mean" attribute.
        value = string.split("mean: ")[1]  # tensor string
        value = cls.string_to_tensor(value)
        setattr(scaler, "mean", value)
        return scaler


class LogTenScaler(Scaler):
    def __init__(self, dim=0):
        """
        Keyword Arguments:
            dim (int): The dimension along which data should be transformed.
        """
        super().__init__(dim)
        self.linear = False

    def transform(self, data):
        data = super().transform(data)
        data = torch.log10(data)
        return data

    def inverse_transform(self, data):
        data = super().inverse_transform(data)
        return 10**data


class ClampedLogTenScaler(Scaler):
    def __init__(self, dim=0):
        """
        Keyword Arguments:
            dim (int): The dimension along which data should be transformed.
        """
        super().__init__(dim)
        self.linear = False
        self.threshold = 30

    def transform(self, data):
        data = super().transform(data)
        data = torch.log10(data)
        return data

    def inverse_transform(self, data):
        data = super().inverse_transform(data)
        data = torch.clamp(
            data,
            min=-self.threshold,
            max=self.threshold,
        )
        return 10**data


class LogTenDeltaScaler(Scaler):
    def __init__(self, delta=1, dim=0):
        """
        Keyword Arguments:
            dim (int): The dimension along which data should be transformed.
        """
        super().__init__(dim)
        self.linear = False
        self.threshold = 30
        self.delta = self.format_tensorlike(delta)

    @classmethod
    def tensorlike_attrs(cls):
        """
        A list of attributes of self that should be tensors.
        """
        return ["delta"]

    def transform(self, data):
        data = super().transform(data)
        data = torch.log10(data + self.delta)
        return data

    def inverse_transform(self, data):
        data = super().inverse_transform(data)
        data = torch.clamp(
            data,
            min=-self.threshold,
            max=self.threshold,
        )
        return (10**data) - self.delta

    @classmethod
    def from_string(cls, string):
        # get "delta"
        value = string.split("delta: ")[1]  # tensor string
        value = cls.string_to_tensor(value)
        return cls(dim=cls.dim_from_string(string), delta=value)


class ClampScaler(Scaler):
    def __init__(self, dim=0):
        """
        Keyword Arguments:
            dim (int): The dimension along which data should be transformed.
        """
        super().__init__(dim)
        self.linear = False
        self.threshold = 30

    def transform(self, data):
        data = super().transform(data)
        return data

    def inverse_transform(self, data):
        data = super().inverse_transform(data)
        data = torch.clamp(
            data,
            min=-self.threshold,
            max=self.threshold,
        )
        return data


class MinMaxScaler(Scaler):
    def __init__(self, dim=0):
        """
        Keyword Arguments:
            dim (int): The dimension along which data should be transformed.
        """
        super().__init__(dim)
        self.linear = True
        self.min = None
        self.max = None

    @classmethod
    def tensorlike_attrs(cls):
        """
        A list of attributes of self that should be tensors.
        """
        return ["min", "max"]

    def fit(self, data):
        data = super().fit(data)
        self.min = torch.clone(data).min(dim=self.dim)[0]
        self.max = torch.clone(data).max(dim=self.dim)[0]

    def transform(self, data):
        data = super().transform(data)
        return (data - self.min.unsqueeze(self.dim)) / (
            self.max.unsqueeze(self.dim) - self.min.unsqueeze(self.dim)
        )

    def inverse_transform(self, data):
        data = super().inverse_transform(data)
        return data * (
            self.max.unsqueeze(self.dim) - self.min.unsqueeze(self.dim)
        ) + self.min.unsqueeze(self.dim)

    @classmethod
    def from_string(cls, string):
        scaler = cls(dim=cls.dim_from_string(string))
        # set the "max" attribute
        max_value = string.split("max: ")[1].split("min: ")[0][
            :-2
        ]  # tensor-like string
        max_value = cls.string_to_tensor(max_value)
        setattr(scaler, "max", max_value)
        # set the "min" attribute
        min_value = string.split("min: ")[1]
        min_value = cls.string_to_tensor(min_value)
        setattr(scaler, "min", min_value)
        return scaler


class ProductScaler(Scaler):
    def __init__(self, multiplier, dim=0):
        """
        Keyword Arguments:
            dim (int): The dimension along which data should be transformed.
        """
        super().__init__(dim)
        self.multiplier = self.format_tensorlike(multiplier)
        self.linear = True

    @classmethod
    def tensorlike_attrs(cls):
        """
        A list of attributes of self that should be tensors.
        """
        return ["multiplier"]

    def transform(self, data):
        data = super().transform(data)
        return data * self.multiplier.unsqueeze(self.dim)

    def inverse_transform(self, data):
        data = super().inverse_transform(data)
        return data / self.multiplier.unsqueeze(self.dim)

    @classmethod
    def from_string(cls, string):
        # get "multiplier" attribute
        value = string.split("multiplier: ")[1]  # tensor-like string
        value = cls.string_to_tensor(value)
        if cls == ProductScaler:
            return cls(
                dim=cls.dim_from_string(string),
                multiplier=value,
            )
        elif cls == QuotientScaler:
            return cls(
                dim=cls.dim_from_string(string),
                divisor=1 / value,
            )


class QuotientScaler(ProductScaler):
    def __init__(self, divisor, dim=0):
        """
        Keyword Arguments:
            dim (int): The dimension along which data should be transformed.
        """
        super().__init__(multiplier=(1 / divisor), dim=dim)


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

    def fit(self, data):
        return data
