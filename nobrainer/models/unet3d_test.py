# -*- coding: utf-8 -*-
"""Tests for MeshNet."""

import numpy as np
import tensorflow as tf

from nobrainer.models.unet3d import UNet3D


def test_unet3d():
    shape = (1, 5, 5, 5)
    X = np.random.rand(*shape, 1).astype(np.float32)
    y = np.random.randint(0, 9, size=(shape), dtype=np.int32)

    def dset_fn():
        return tf.data.Dataset.from_tensors((X, y))

    estimator = UNet3D(
        n_classes=10,
        optimizer='Adam',
        learning_rate=0.001)
    estimator.train(input_fn=dset_fn)

    # With optimizer object.
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
    estimator = UNet3D(
        n_classes=10,
        optimizer=optimizer,
        learning_rate=0.001)
    estimator.train(input_fn=dset_fn)
