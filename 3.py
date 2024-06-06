import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

class CNN():
    """
    A simple Convolutional Neural Network (CNN) class.

    Attributes
    ----------
    conv1 : Conv2D
        The first convolutional layer with 32 filters, a 3x3 kernel size, 'relu' activation function and an input shape of (28, 28, 1).
    conv2 : Conv2D
        The second convolutional layer with 64 filters, a 3x3 kernel size and 'relu' activation function.
    pool : MaxPooling2D
        The max pooling layer with a pool size of 2x2.
    flat : Flatten
        The flatten layer to convert the 2D matrix data into a vector.
    dense1 : Dense
        The first dense (fully connected) layer with 128 neurons and 'relu' activation function.
    dense2 : Dense
        The second dense (fully connected) layer with 10 neurons and 'softmax' activation function.
    """

    def __init__(self):
        """
        The constructor for CNN class.
        """
        self.conv1 = Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))
        self.conv2 = Conv2D(64, (3, 3), activation='relu')
        self.pool = MaxPooling2D(pool_size=(2, 2))
        self.flat = Flatten()
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(10, activation='softmax')

    def forward(self, x):
        """
        The forward propagation method for CNN class.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        x : Tensor
            The output tensor after passing through the network.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

if __name__ == '__main__':
    cnn = CNN()
    x = tf.random.normal((1, 28, 28, 1))
    print(cnn.forward(x).shape)