from keras.activations import sigmoid, softmax
from keras.layers import Layer
from tensorflow import concat


class YoloActivation(Layer):
    def call(self, inputs):
        return concat([
            sigmoid(inputs[..., 0:10]),
            softmax(inputs[..., 10:]),
        ], axis=-1)
