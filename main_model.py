import numpy as np
from keras.layers import Input, Dense
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Concatenate

from custom_nets import CoordinatesNet, AnglesNet, SphericalHarmonicsNet, FourierTransformTensor
from custom_nets import FourierTransformNet, ConcatNets, FinalLayer

# encoder_dims/deepconcat_dims are the number of neurons in each hidden layer. Can provide 1 to 3 values resulting in 1 to 3 layers

n_neighbours = 10 # number of nearest neighbours in input
len_input = n_neighbours * 3 # times 3 as xyz

# Input tensor order is x1, y1, z1, x2, y2,z2, ..., x10, y10, z10
# where number corresponds to nearest neighbour ID
inputTensor = Input(shape=(len_input,))

# Coordinates Network
DeepCoords = CoordinatesNet(inputTensor, n_neighbours, encoder_dims=[250, 50], deepconcat_dims=[250, 50])

# Fourier Transform Network
fft_x, fft_y, fft_z = FourierTransformTensor(inputTensor, n_neighbours, nbins = 30, ValueRange=[-6.0, 6.0])
DeepFFT = FourierTransformNet(fft_x, fft_y, fft_z, encoder_dims=[250, 50], deepconcat_dims=[250, 50])

# Spherical Coordinates Network
DeepAngles = AnglesNet(inputTensor, len_input, n_neighbours, encoder_dims=[250, 50], deepconcat_dims=[250, 50])

# Spherical Harmonics Network
DeepY2 = SphericalHarmonicsNet(inputTensor, len_input, n_neighbours, spherical_ID=2, encoder_dims=[250, 50], deepconcat_dims=[250, 50])
DeepY3 = SphericalHarmonicsNet(inputTensor, len_input, n_neighbours, spherical_ID=3, encoder_dims=[250, 50], deepconcat_dims=[250, 50])
DeepY4 = SphericalHarmonicsNet(inputTensor, len_input, n_neighbours, spherical_ID=4, encoder_dims=[250, 50], deepconcat_dims=[250, 50])
YCat = Concatenate()([DeepY2, DeepY3, DeepY4])
DeepYCat_h1 = Dense(200, activation='relu')(YCat)
DeepY = Dense(50, activation='relu')(DeepYCat_h1)


# FinalCat = ConcatNets(DeepCoords)
FinalCat = ConcatNets(DeepFFT, DeepCoords, DeepAngles, DeepY)

#output_layer = FinalLayer(FinalCat, skip=True, num_classes=2)
output_layer = FinalLayer(FinalCat, hidden_layers =[250, 50, 5], skip=False, num_classes=2)

model = Model(inputTensor, output_layer)

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function   categorical_crossentropy
                  optimizer="adam", # using the Adam optimiser adam
                  metrics=['accuracy']) # reporting the accuracy



data = np.load("input_rotated_1D_3loops_TrainTest.npz")  #
data = np.load("input_normal_1D_TrainTest.npz")  #

X_train, y_train = data['X_train'], data['y_train']

model.fit(X_train, y_train, batch_size=2000, epochs=10,verbose=2, validation_split=0.1)

X_test, y_test = data['X_test'], data['y_test']
model.evaluate(X_test, y_test, verbose=0)

model.save_weights('model_spherical_1_weights.h')
model.load_weights('model_spherical_1_weights.h')

