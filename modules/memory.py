from abc import abstractmethod
from typing import Tuple, Dict

import numpy as np
import torch


class Buffer:
    def __init__(self, buffer_size, device="cpu"):
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.attributes = ["state", "action", "reward", "next_state", "mask"]
        self.state: torch.Tensor
        self.action: torch.Tensor
        self.reward: torch.Tensor
        self.next_state: torch.Tensor
        self.mask: torch.Tensor
        self.is_init: bool = False

    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self

    def init_tensors(self, **kwargs) -> None:
        """Initialize tensors based on input shape"""

        for attr_str in self.attributes:
            attr = kwargs[attr_str]
            assert attr is not None, f"Missing attribute {attr}"
            assert not hasattr(self, attr_str), "Can not initialize twice."
            typ = torch.int64 if attr_str in ["action", "mask"] else torch.float32
            setattr(
                self,
                attr_str,
                torch.zeros(
                    (self.buffer_size, *attr.shape[1:]), dtype=typ, device=self.device
                ),
            )
        self.is_init = True

    def reset_tensors(self) -> None:
        for attr_str in self.attributes:
            setattr(self, attr_str, torch.zeros_like(getattr(self, attr_str)))

    def add_data(self, **kwargs):
        if not self.is_init:
            self.init_tensors(**kwargs)
        return self._add_data(**kwargs)

    @abstractmethod
    def _add_data(self, **kwargs):
        pass

    def get_data(self, size: int, random=True, return_index=False) -> Dict:
        if size > min(self.num_seen_examples, self.state.shape[0]):
            size = min(self.num_seen_examples, self.state.shape[0])
        if random:
            choice = np.random.choice(
                min(self.num_seen_examples, self.state.shape[0]), size=size, replace=False
            )

        return_dict = {}
        for attr_str in self.attributes:

            attr = getattr(self, attr_str)
            if random:
                attr = attr[choice]
            return_dict[attr_str] = attr

        if return_index:
            return_dict["index"] = torch.tensor(choice).to(self.device)
        return return_dict

    def get_all_data(self) -> Tuple:
        """
        Return all the items in the memory buffer.
        """
        return self.get_data(len(self), random=False, return_index=False)

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def is_full(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples >= self.buffer_size:
            return True
        else:
            return False

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        self.reset_tensors()
        self.num_seen_examples = 0

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)


class ReservoirBuffer(Buffer):
    """
    Random Buffer
    """

    def __init__(self, buffer_size, device="cpu"):
        super().__init__(buffer_size, device)

    def _replace(self, input, dest_index, dest_name, p):
        end_idx = int(min(self.buffer_size * p, input.shape[0], len(dest_index)))
        dest_index = dest_index[:end_idx]
        rand_src = torch.randperm(input.shape[0])[:end_idx]
        attr = getattr(self, dest_name)
        attr[dest_index] = input[rand_src].to(self.device)
        return len(dest_index)

    def _append(self, input, dest_name, p):
        capacity = self.buffer_size - len(self)
        assert capacity > 0
        seq_dest = torch.arange(self.buffer_size)[-capacity:]
        n_added = self._replace(input, seq_dest, dest_name, p)
        self.num_seen_examples = min(self.buffer_size, self.num_seen_examples + n_added)
        return n_added

    def _random_replace(self, input, dest_name, p):

        rand_dest = torch.randperm(len(self))
        return self._replace(input, rand_dest, dest_name, p)

    def _add_attr(self, input, dest_name, p=0.2):
        """
        append data until buffer is full
        """
        capacity = self.buffer_size - len(self)
        if capacity > 0:
            # there is enough capacity. add the samples sequentially
            self._append(input, dest_name, p=p)

        else:
            # buffer is full
            self._random_replace(input, dest_name, p=p)

    def _add_data(self, p=0.2, **kwargs):
        """
        replaces at most `p` of the buffer with the added samples
        """
        for attr_str in self.attributes:
            self._add_attr(kwargs[attr_str], attr_str, p=p)
