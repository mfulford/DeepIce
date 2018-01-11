from keras.engine.topology import Layer
from keras import backend as K

""" Layer that converts 1d vector input into a histogram 

    Input is the x, y or z coordinates of the n nearest neighbours
    Returns the histogram of the n nearest neighbour coordinates with nbins within the range defined 
    by ValueRange

    # Example
    hist_x = Layer_Hist(nbins=30, ValueRange=[-6.0, 6.0], input_shape=(10,))(x)

"""


class Layer_Hist(Layer):

    def __init__(self, nbins=30, ValueRange=[-6.0, 6.0], **kwargs):
        self.output_dim = nbins
        self.ValueRange = ValueRange
        self.nbins = nbins
        self.binwidth = abs((self.ValueRange[1] - self.ValueRange[0])/(self.nbins-2))
        super(Layer_Hist, self).__init__(**kwargs)

    def call(self, x):

        y = x[1]
        BinPop = {}

        id_bin_1 = K.cast(K.less(x, self.ValueRange[0]), K.floatx())
        id_bin_last = K.cast(K.greater_equal(x, self.ValueRange[1]), K.floatx())

        BinPop[1] = K.sum(id_bin_1, axis=1, keepdims=True)
        BinPop[self.nbins] = K.sum(id_bin_last, axis=1, keepdims=True)

        start = self.ValueRange[0]
        stop = start + self.binwidth
        for i in range(2, self.nbins):

            id_bin = K.tf.multiply(K.cast(K.greater_equal(x, start), K.floatx()),
                                  K.cast(K.less(x, stop), K.floatx()))
            BinPop[i] = K.sum(id_bin, axis=1, keepdims=True)

            for i2 in range(2, self.nbins):
                id_bin_i2 = K.tf.multiply(K.cast(K.greater_equal(y, start), K.floatx()),
                                  K.cast(K.less(y, stop), K.floatx()))

            start = stop
            stop = start + self.binwidth

        hist = K.concatenate(list(BinPop.values()))
        return hist

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
