from abc import abstractmethod
from typing import Tuple, Dict

import numpy as np
import torch


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


class Buffer:
    def __init__(self, buffer_size, device='cpu'):
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.attributes = ['states', 'actions', 'rewards', 'next_states', 'masks']

    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self

    def init_tensors(self, states: torch.Tensor, actions: torch.Tensor,
                     rewards: torch.Tensor, next_states: torch.Tensor,
                     masks: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str in ['masks'] else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                                                     *attr.shape[1:]), dtype=typ, device=self.device))

    @abstractmethod
    def add_data(self, states: torch.Tensor, actions: torch.Tensor,
                 rewards: torch.Tensor, next_states: torch.Tensor,
                 masks: torch.Tensor):
        pass

    def get_data(self, size: int, return_index=False) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.states.shape[0]):
            size = min(self.num_seen_examples, self.states.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.states.shape[0]),
                                  size=size, replace=False)

        ret_tuple = {}
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple[attr_str] = attr[choice]

        if return_index:
            ret_tuple['index'] = torch.tensor(choice).to(self.device)
        return ret_tuple
    
    def get_all_data(self):
        return self.get_data(size=len(self))

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
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)


class ReservoirBuffer(Buffer):
    def __init__(self, config, device='cpu'):
        super().__init__(config, device)

    def init_tensors(self, states: torch.Tensor, actions: torch.Tensor,
                     rewards: torch.Tensor, next_states: torch.Tensor,
                     masks: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str in ['masks'] else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size, *attr.shape[1:]), dtype=typ, device=self.device))

    def add_data(self, states: torch.Tensor, actions: torch.Tensor,
                 rewards: torch.Tensor, next_states: torch.Tensor,
                 masks: torch.Tensor):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        """
        if not hasattr(self, 'x'):
            self.init_tensors(states, actions, rewards, next_states, masks)

        added = 0

        for i in range(states.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.states[index] = states[i].to(self.device)
                self.actions[index] = actions[i].to(self.device)
                self.rewards[index] = rewards[i].to(self.device)
                self.next_states[index] = next_states[i].to(self.device)
                self.masks[index] = masks[i].to(self.device)
                added += 1
        return added


class FIFOBuffer(Buffer):
    def __init__(self, config, device='cpu'):
        super().__init__(config, device)
        self.empty()

    def add_data(self, states: torch.Tensor, actions: torch.Tensor,
                 rewards: torch.Tensor, next_states: torch.Tensor,
                 masks: torch.Tensor):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        """

        for i in range(states.shape[0]):
            self.num_seen_examples += 1
            self.states.append(states[i].to(self.device))
            self.actions.append(actions[i].to(self.device))
            self.rewards.append(rewards[i].to(self.device))
            self.next_states.append(next_states[i].to(self.device))
            self.masks.append(masks[i].to(self.device))

        if len(self.states) > 2 * self.buffer_size:
            for attr_str in self.attributes:
                attr = getattr(self, attr_str)
                setattr(self, attr_str, attr[-self.buffer_size:])

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            setattr(self, attr_str, [])
        self.num_seen_examples = 0

    def get_data(self, size: int, return_index=False, recent=None) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        recent_size = 0
        if recent is not None:
            assert recent >= 0
            recent_size = min(self.num_seen_examples, self.buffer_size, recent)

        if size > min(self.num_seen_examples, self.buffer_size):
            size = min(self.num_seen_examples, self.buffer_size)
        size -= recent_size

        choice = []
        # retrieve recent samples
        if recent_size > 0:
            choice.extend(list(np.arange(len(self))[-recent_size:]))
        # retrieve random samples
        if size > 0:
            select_range = len(self) if recent_size == 0 else len(self) - recent_size
            choice.extend(list(np.random.choice(select_range, size=size, replace=False)))

        ret_tuple = {}
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)[-self.buffer_size:]
                ret_tuple[attr_str] = torch.stack([attr[c] for c in choice])

        if return_index:
            ret_tuple['index'] = torch.tensor(choice).to(self.device)
        return ret_tuple


class PrioritizedFIFOBuffer(Buffer):
    def __init__(self, config, device='cpu'):
        super().__init__(config, device)
        self.store = 0

    def step(self):
        self.store += 1
        if self.store >= self.buffer_size:
            self.store = 0

    def add_data(self, states: torch.Tensor, actions: torch.Tensor,
                 rewards: torch.Tensor, next_states: torch.Tensor,
                 masks: torch.Tensor):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        """
        if not hasattr(self, 'x'):
            self.init_tensors(states, actions, rewards, next_states, masks)

        for i in range(states.shape[0]):
            index = self.store
            self.num_seen_examples += 1
            if np.random.rand() > (self.rewards[index] / rewards[i].item()):
                self.states[index] = states[i].to(self.device)
                self.actions[index] = actions[i].to(self.device)
                self.rewards[index] = rewards[i].to(self.device)
                self.next_states[index] = next_states[i].to(self.device)
                self.masks[index] = masks[i].to(self.device)
            self.step()


class PDBuffer(ReservoirBuffer):
    """
    Random Buffer
    """

    def __init__(self, config, seq_len, device='cpu'):
        super().__init__(config, device)
        self.seq_len = seq_len

    def init_tensors(self, states: torch.Tensor, actions: torch.Tensor,
                     rewards: torch.Tensor, next_states: torch.Tensor,
                     masks: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str in ['masks'] else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size, self.seq_len,
                                                     *attr.shape[1:]), dtype=typ, device=self.device))
