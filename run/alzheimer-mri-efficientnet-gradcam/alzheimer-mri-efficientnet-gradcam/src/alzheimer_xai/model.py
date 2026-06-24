from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    Input,
    RandomFlip,
    RandomRotation,
    Dense,
    Flatten,
    BatchNormalization,
    Dropout,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers


def build_efficientnet_model(
    num_classes: int = 4,
    image_size: tuple[int, int] = (224, 224),
    learning_rate: float = 0.001,
    fine_tune_from_layer: str = "block5a_expand_activation",
) -> Model:
    """Build EfficientNetB0 classifier following the original notebook."""
    inputs = Input(shape=(image_size[0], image_size[1], 3))
    x = RandomFlip("horizontal")(inputs)
    x = RandomRotation(0.1)(x)

    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )

    backbone.trainable = True
    set_trainable = False
    for layer in backbone.layers:
        if layer.name == fine_tune_from_layer:
            set_trainable = True
        layer.trainable = set_trainable

    x = backbone.output
    x = Flatten()(x)
    x = Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
