from typing import Tuple

from keras.layers import (Conv2D, Dense, Flatten, InputLayer, LeakyReLU,
                          MaxPooling2D, Reshape)
from keras.models import Sequential

from yolo.activation import YoloActivation


def GetModel(input_shape: Tuple[int, int, int]) -> Sequential:
    return Sequential(
        layers=[
            InputLayer(input_shape=input_shape),
            Conv2D(64, (7, 7), strides=(2, 2),
                   padding="same"), LeakyReLU(alpha=0.1),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

            Conv2D(192, (3, 3), padding="same"), LeakyReLU(alpha=0.1),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

            Conv2D(128, (1, 1), padding="same"), LeakyReLU(alpha=0.1),
            Conv2D(256, (3, 3), padding="same"), LeakyReLU(alpha=0.1),
            Conv2D(256, (1, 1), padding="same"), LeakyReLU(alpha=0.1),
            Conv2D(512, (3, 3), padding="same"), LeakyReLU(alpha=0.1),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

            Conv2D(256, (1, 1), padding="same"), LeakyReLU(alpha=0.1),
            Conv2D(512, (3, 3), padding="same"), LeakyReLU(alpha=0.1),
            Conv2D(256, (1, 1), padding="same"), LeakyReLU(alpha=0.1),
            Conv2D(512, (3, 3), padding="same"), LeakyReLU(alpha=0.1),
            Conv2D(256, (1, 1), padding="same"), LeakyReLU(alpha=0.1),
            Conv2D(512, (3, 3), padding="same"), LeakyReLU(alpha=0.1),
            Conv2D(256, (1, 1), padding="same"), LeakyReLU(alpha=0.1),
            Conv2D(512, (3, 3), padding="same"), LeakyReLU(alpha=0.1),
            Conv2D(512, (1, 1), padding="same"), LeakyReLU(alpha=0.1),
            Conv2D(1024, (3, 3), padding="same"), LeakyReLU(alpha=0.1),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

            Conv2D(512, (1, 1), padding="same"), LeakyReLU(alpha=0.1),
            Conv2D(1024, (3, 3), padding="same"), LeakyReLU(alpha=0.1),
            Conv2D(512, (1, 1), padding="same"), LeakyReLU(alpha=0.1),
            Conv2D(1024, (3, 3), padding="same"), LeakyReLU(alpha=0.1),
            Conv2D(1024, (3, 3), padding="same"), LeakyReLU(alpha=0.1),
            Conv2D(1024, (3, 3), strides=(2, 2),
                   padding="same"), LeakyReLU(alpha=0.1),

            Conv2D(1024, (3, 3), padding="same"), LeakyReLU(alpha=0.1),
            Conv2D(1024, (3, 3), padding="same"), LeakyReLU(alpha=0.1),

            Flatten(),
            Dense(4096), LeakyReLU(alpha=0.1),

            Dense(7 * 7 * 30),

            Reshape((7, 7, 30)),
            YoloActivation(),
        ]
    )
