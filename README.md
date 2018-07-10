# Keras Switchable Normalization

Switchable Normalization is a normalization technique that is able to learn different normalization operations for different normalization layers in a deep neural network in an end-to-end manner.

Keras port of the implementation of the paper [Differentiable Learning-to-Normalize via Switchable Normalization](https://arxiv.org/abs/1806.10779).

Code ported from [the switchnorm official repository](https://github.com/switchablenorms/Switchable-Normalization).

**Note**

This only implements the moving average version of batch normalization component from the paper. The `batch average` technique cannot be easily implemented in Keras as a layer, and therefore it is not supported.

# Usage

Simply import `switchnorm.py` and replace BatchNormalization layer with this layer.

```python
from switchnorm import SwitchNormalization

ip = Input(...)
...
x = SwitchNormalization(axis=-1)(x)
...

```

# Example

An example script for CIFAR 10 is provided in `cifar.py`, which trains a toy model for 25 epochs. It obtains an accuracy of 88%.

## **Issues**

Currently, when training this toy model, near the 22-25th epoch, training abruptly fluctuates and accuracy plummets. Whether this is due to the normalization, or whether this is due to the model itself is being investigated.

# Tests

The tests require `pytest` to be installed and can be run using `pytest .` at the root of the directory. These tests are adapted from the Batch Normalization tests in Keras.

# Requirements

- Keras 2.1.6+
- Either Tensorflow, Theano or CNTK.
