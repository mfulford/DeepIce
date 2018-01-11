from keras.engine.topology import Layer
from keras import backend as K

""" Layer to calculate the Fourier transform 

    Input is the x, y or z coordinates histogram calcualted using Layer_Hist
    Returns: fourier transform of the coordinate histogram. The real and
    imaginary components are concatenated together and returned as a single real vector

    Options: fftDim = 1 returns the 1D Fourier transform. Input is 1D histogram
             fftDim = 2 returns the 2D Fourier transform. Input is a 2D histogram 

    # Example
    fft_x = Layer_FFT(fftDim=1, input_shape=(30,))(hist_x)

"""


class Layer_FFT(Layer):

    def __init__(self, input_shape, fftDim=1, **kwargs):
        if fftDim == 1:
            fft_shape = divmod(input_shape[0],2)[0] + 1
        elif fftDim == 2:
            fft_shape = divmod(input_shape[1], 2)[0] + 1
        self.output_dim = fft_shape * fftDim * 2 # *2 as concat real and imaginary
        self.fftDim = fftDim
        super(Layer_FFT, self).__init__(**kwargs)

    def call(self, hist):

        if self.fftDim == 1:
            fft = K.tf.spectral.rfft(hist)
            fft_real = K.tf.real(fft)
            fft_imag = K.tf.imag(fft)
        elif self.fftDim == 2:
            fft = K.tf.spectral.rfft2d(hist)
            out_shape = fft.shape[1]* fft.shape[2]
            fft = K.reshape(fft, shape=((K.shape(fft)[0], out_shape,)))
            fft_real = K.tf.real(fft)
            fft_imag = K.tf.imag(fft)

        fft_cat = K.concatenate([fft_real, fft_imag], axis=1)

        return fft_cat

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
