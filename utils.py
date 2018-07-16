import tensorflow as tf
import numpy as np


def print_tensors(obj):
    for variable_name, tensor in vars(obj).items():
        if isinstance(tensor, tf.Tensor):
            print('{}:\t({} Shape={})'.format(variable_name, tensor.name, tensor.shape))


def discount_rewards(rewards, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards
