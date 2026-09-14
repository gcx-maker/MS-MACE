import collections
import copy
import re
from typing import Any

import torch


def size_repr(key: str, item: Any, indent: int=0) -> str:
    indent_str = " " * indent
    if torch.is_tensor(item) and item.dim() == 0:
        out = item.item()
    elif torch.is_tensor(item):
        out = str(list(item.shape))
    elif isinstance(item, list) or isinstance(item, tuple):
        out = str(len(item))
    elif isinstance(item, dict):
        lines = [indent_str + size_repr(k, v, indent=2) for k, v in item.items()]
        out = "{\n" + ",\n".join(lines) + "\n" + indent_str + "}"
    elif isinstance(item, str):
        out = f"{item}"
    else:
        out = str(item)

    return f"{indent_str}{key}={out}"


class Data(object):
    """
    Args:
        x (Tensor, optional): Node feature matrix with shape :obj:`[num_nodes,
            num_node_features]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Graph connectivity in COO format
            with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        y (Tensor, optional): Graph or node targets with arbitrary shape.
            (default: :obj:`None`)
        pos (Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
    """
    def __init__(
            self,
            x=None,
            edge_index=None,
            edge_attr=None,
            y=None,
            pos=None,
            **kwargs,
    ):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.pos = pos
        for key, item in kwargs.items():
            if key == "num_nodes":
                self.__num_nodes__ = item
            else:
                self[key] = item

        if edge_index is not None and edge_index.dtype != torch.long:
            raise ValueError(
                (
                    f"Argument 'edge_index' needs to be of type 'torch.long' but "
                    f"found type '{edge_index.dtype}'."
                )
            )

    @classmethod
    def from_dict(cls, dictionary:dict):
        data = cls()

        for key, item in dictionary.items():
            data[key] = item

        return data

    def to_dict(self):
        return {key: item for key, item in self}

    def to_namedtuple(self):
        keys = self.keys
        DataTuple = collections.namedtuple("DataTuple", keys)
        return DataTuple(*[self[key] for key in keys])

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        return delattr(self, key)

    @property
    def keys(self):
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != "__" and key[-2:] != "__"]
        return keys

    def __len__(self):
        return len(self.keys)

    def __contains__(self, key):
        return key in self.keys

    def __iter__(self):
        for key in sorted(self.keys):
            yield key, self[key]

    def __call__(self, *keys):
        for key in sorted(self.keys) if not keys else keys:
            if key in self:
                yield key, self[key]

    @staticmethod
    def __cat_dim__(key, value):
        if bool(re.search("index", key)):
            return -1
        return 0

    def __inc__(self, key, value):
        return self.num_nodes if bool(re.search("index", key)) else 0

    @property
    def num_nodes(self):
        if hasattr(self, "__num_nodes__"):
            return self.__num_nodes__
        for key, item in self("x", "pos", "batch"):
            return item.size(self.__cat_dim__(key, item))
        return None

    @num_nodes.setter
    def num_nodes(self, num_nodes):
        self.__num_nodes__ = num_nodes

    @property
    def num_edges(self):
        for key, item in self("edge_index", "edge_attr"):
            return item.size(self.__cat_dim__(key, item))
        return None

    @property
    def num_node_features(self):
        if self.x is None:
            return 0
        return 1 if self.x.dim() == 1 else self.x.size(-1)

    @property
    def num_features(self):
        return self.num_node_features

    def __apply__(self, item, func):
        if torch.is_tensor(item):
            return func(item)
        elif isinstance(item, tuple):
            return tuple(
            self.__apply__(v, func)
            for v in item
            )
        elif isinstance(item, list):
            return [
                self.__apply__(v, func)
                for v in item
            ]
        elif isinstance(item, dict):
            return {k: self.__apply__(v, func) for k, v in item.items()}
        else:
            return item

    def apply(self, func, *keys):
        for key, item in self(*keys):
            self[key] = self.__apply__(item, func)
        return self

    def contiguous(self, *keys):
        return self.apply(lambda x: x.contiguous(), *keys)

    def to(self, device, *keys, **kwargs):
        return self.apply(lambda x: x.to(device, **kwargs), *keys)

    def cpu(self, *keys):
        return self.apply(lambda x: x.cpu(), *keys)

    def cuda(self, device=None, non_blocking=False, *keys):
        return self.apply(
            lambda x: x.cuda(device=device, non_blocking=non_blocking), *keys
        )

    def clone(self):
        return self.__class__.from_dict(
            {
                k: v.clone() if torch.is_tensor(v) else copy.deepcopy(v)
                for k, v in self.__dict__.items()
            }
        )

    def pin_memory(self, *keys):
        return self.apply(lambda x: x.pin_memory(), *keys)

    def debug(self):
        if self.edge_index is not None:
            if self.edge_index.dtype != torch.long:
                raise RuntimeError(
                    (
                        "Expected edge indices of dtype {}, but found dtype " " {}"
                    ).format(torch.long, self.edge_index.dtype)
                )
        if self.edge_index is not None:
            if self.edge_index.dim() != 2 or self.edge_index.size(0) != 2:
                raise RuntimeError(
                    (
                        "Edge indices should have shape [2, num_edges] but found"
                        " shape {}"
                    ).format(self.edge_index.size())
                )
        if self.edge_index is not None and self.num_nodes is not None:
            if self.edge_index.numel() > 0:
                min_index = self.edge_index.min()
                max_index = self.edge_index.max()
            else:
                min_index = max_index = 0
            if min_index < 0 or max_index > self.num_nodes - 1:
                raise RuntimeError(
                    (
                        "Edge indices must lay in the interval [0, {}]"
                        " but found them in the interval [{}, {}]"
                    ).format(self.num_nodes - 1, min_index, max_index)
                )
        if self.edge_index is not None and self.edge_attr is not None:
            if self.edge_index.size(1) != self.edge_attr.size(0):
                raise RuntimeError(
                    (
                        "Edge indices and edge attributes hold a differing "
                        "number of edges, found {} and {}"
                    ).format(self.edge_index.size(), self.edge_attr.size())
                )

        if self.x is not None and self.num_nodes is not None:
            if self.x.size(0) != self.num_nodes:
                raise RuntimeError(
                    (
                        "Node features should hold {} elements in the first "
                        "dimension but found {}"
                    ).format(self.num_nodes, self.x.size(0))
                )

        if self.pos is not None and self.num_nodes is not None:
            if self.pos.size(0) != self.num_nodes:
                raise RuntimeError(
                    (
                        "Node positions should hold {} elements in the first "
                        "dimension but found {}"
                    ).format(self.num_nodes, self.pos.size(0))
                )
    def __repr__(self):
        cls = str(self.__class__.__name__)
        has_dict = any([isinstance(item, dict) for _, item in self])

        if not has_dict:
            info = [size_repr(key, item) for key, item in self]
            return "{}({})".format(cls, ", ".join(info))
        else:
            info = [size_repr(key, item, indent=2) for key, item in self]
            return "{}(\n{}\n)".format(cls, ",\n".join(info))



# data = Data(
#     pos=torch.randn(5, 3),
#     edge_index=torch.tensor(
#         [
#             [0, 1, 2, 3],
#             [1, 2, 3, 4]
#         ],
#         dtype=torch.long
#     ),
#     energy=torch.tensor(-123.456),
#     z=torch.tensor([1, 6, 8, 1, 1]),
# )
#
# print(data)
#
# data = Data(
#     pos=torch.randn(5,3),
#     edge_index=torch.tensor(
#         [[0,1,2],
#          [1,2,3]],
#         dtype=torch.long
#     ),
#     stats={
#         "energy_mean": torch.tensor(-10.5),
#         "energy_std": torch.tensor(2.3)
#     }
# )
#
# print(data)