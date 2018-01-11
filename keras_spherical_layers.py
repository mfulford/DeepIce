import numpy as np
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model # basic class for specifying and training a neural network
from keras.models import load_model
from keras.layers import Concatenate, Lambda

from histogram_layer import Layer_Hist
from angle_layer import Layer_Angles
from spherical_layer import Layer_Y
from fft_layer import Layer_FFT

# Edit shared weights 9540s  - loss: 0.0593 - acc: 0.9790 - val_loss: 0.0615 - val_acc: 0.9783  108,042 params
# Shared weights v2: 13195s -  loss: 0.0903 - acc: 0.9671 - val_loss: 0.0891 - val_acc: 0.9674  604,682 params
# fft1:              15608s -  loss: 0.0609 - acc: 0.9784 - val_loss: 0.0640 - val_acc: 0.9773  966,282 params - fft of xyz coords.  + shared weights
# fft2: 9212s -                loss: 0.0538 - acc: 0.9810 - val_loss: 0.0570 - val_acc: 0.9798  716,462 params - with spherical fft
# fft3:  - no spherical fft. + fft of x/y/z separately. Then combined in deep Cat.
# 9365s - loss: 0.0660 - acc: 0.9763 - val_loss: 0.0692 - val_acc: 0.9751
# fft4: 5527s - loss: 0.0791 - acc: 0.9715 - val_loss: 0.0782 - val_acc: 0.9718
# Original 5336s - loss: 0.0397 - acc: 0.9861 - val_loss: 0.0565 - val_acc: 0.9805


data = np.load("input_rotated_1D_3loops_TrainTest.npz")  #
data = np.load("input_normal_1D_TrainTest.npz")  #

X_train, y_train = data['X_train'], data['y_train']
#X_test, y_test = data['X_test'], data['y_test']

n_neighbours = 10 # number of nearest neighbours in input
len_input = n_neighbours * 3 # times 3 as xyz

# Input tensor order is x1, y1, z1, x2, y2,z2, ..., x10, y10, z10
# where number corresponds to nearest neighbour ID
inputTensor = Input(shape=(len_input,))

CoordinateEncoder_1 = Dense(250, activation="relu") #100 (250)
CoordinateEncoder_2 = Dense(250, activation="relu") #200  (250)
CoordinateEncoder_3 = Dense(50, activation="relu") #50

EncodedCoords = {}
for i in range(1,n_neighbours+1):
    id_start = (i-1)*3
    id_end = id_start + 3
    xyz = Lambda(lambda x: x[:, id_start:id_end], output_shape=((3,)))(inputTensor)
    encoded_h1 = CoordinateEncoder_1(xyz)
    encoded_h2 = CoordinateEncoder_2(encoded_h1)
    EncodedCoords[i] = CoordinateEncoder_3(encoded_h2)

EncodedCoordsCat = Concatenate()(list(EncodedCoords.values()))
Encoded_CatCoords_1 = Dense(250, activation="relu")(EncodedCoordsCat) #500
DeepCoords = Dense(50, activation="relu")(Encoded_CatCoords_1)


# indices of x, y and z coordinates within inputTensor
id_x = K.tf.constant([(x-1)*3 for x in range(1,n_neighbours+1)])
id_y = K.tf.constant([(x-1)*3 +1 for x in range(1,n_neighbours+1)])
id_z = K.tf.constant([(x-1)*3 +2 for x in range(1,n_neighbours+1)])

# x, y, z coordinates as 3 individual tensors
x_tensor = Lambda(lambda x, indices: K.tf.gather(x, indices, axis=1), output_shape=((n_neighbours,)), arguments={'indices': id_x})(inputTensor)
y_tensor = Lambda(lambda x, indices: K.tf.gather(x, indices, axis=1), output_shape=((n_neighbours,)), arguments={'indices': id_y})(inputTensor)
z_tensor = Lambda(lambda x, indices: K.tf.gather(x, indices, axis=1), output_shape=((n_neighbours,)), arguments={'indices': id_z})(inputTensor)

# 1D histograms of coordinates
nbins = 30
hist_x = Layer_Hist(nbins=nbins, ValueRange=[-6.0, 6.0], input_shape=(n_neighbours,))(x_tensor)
hist_y = Layer_Hist(nbins=nbins, ValueRange=[-6.0, 6.0], input_shape=(n_neighbours,))(y_tensor)
hist_z = Layer_Hist(nbins=nbins, ValueRange=[-6.0, 6.0], input_shape=(n_neighbours,))(z_tensor)

# 1D Fourier Transform of 1D histograms
fft_x = Layer_FFT(fftDim=1, input_shape=(nbins,))(hist_x)
fft_y = Layer_FFT(fftDim=1, input_shape=(nbins,))(hist_y)
fft_z = Layer_FFT(fftDim=1, input_shape=(nbins,))(hist_z)

FFT_Encoder_1D_1 = Dense(100, activation="relu")
FFT_Encoder_1D_2 = Dense(20, activation="relu")

fft_1d_x_1 = FFT_Encoder_1D_1(fft_x)
fft_1d_z_1 = FFT_Encoder_1D_1(fft_y)
fft_1d_y_1 = FFT_Encoder_1D_1(fft_z)
fft_1d_x_2 = FFT_Encoder_1D_2(fft_1d_x_1)
fft_1d_y_2 = FFT_Encoder_1D_2(fft_1d_y_1)
fft_1d_z_2 = FFT_Encoder_1D_2(fft_1d_z_1)

Cat_fft_1d = Concatenate()([fft_1d_x_2, fft_1d_y_2, fft_1d_z_2])
Encoded_CatFFT_1d_1 = Dense(100, activation="relu")(Cat_fft_1d)
DeepFFT = Dense(50, activation="relu")(Encoded_CatFFT_1d_1)


angles = Layer_Angles(input_shape=(len_input,))(inputTensor)
y2 = Layer_Y(spherical_ID=2, input_shape=(len_input,))(inputTensor)
y3 = Layer_Y(spherical_ID=3, input_shape=(len_input,))(inputTensor)
y4 = Layer_Y(spherical_ID=4, input_shape=(len_input,))(inputTensor)


AnglesEncoder_h1 = Dense(250, activation="relu")
AnglesEncoder_h2 = Dense(250, activation="relu")
AnglesEncoder_h3 = Dense(50, activation="relu")

Y2Encoder_h1 = Dense(100, activation="relu")
Y2Encoder_h2 = Dense(10, activation="relu")

Y3Encoder_h1 = Dense(100, activation="relu")
Y3Encoder_h2 = Dense(10, activation="relu")

Y4Encoder_h1 = Dense(100, activation="relu")
Y4Encoder_h2 = Dense(10, activation="relu")

EncodedAngles = {}
EncodedY2 = {}
EncodedY3 = {}
EncodedY4 = {}
for i in range(1,11):
    id_start = (i-1)*3
    id_end = id_start + 3
    inp = Lambda(lambda x: x[:, id_start:id_end], output_shape=((3,)))(angles)
    encodedAngles_h1 = AnglesEncoder_h1(inp)
    encodedAngles_h2 = AnglesEncoder_h2(encodedAngles_h1)
    EncodedAngles[i] = AnglesEncoder_h3(encodedAngles_h2)

    id_start = (i-1)*5
    id_end = id_start + 5
    inp = Lambda(lambda x: x[:, id_start:id_end], output_shape=((5,)))(y2)
    encodedY2_h1    = Y2Encoder_h1(inp)
    EncodedY2[i] = Y2Encoder_h2(encodedY2_h1)

    id_start = (i - 1) * 7
    id_end = id_start + 7
    inp = Lambda(lambda x: x[:, id_start:id_end], output_shape=((7,)))(y3)
    encodedY3_h1 = Y3Encoder_h1(inp)
    EncodedY3[i] = Y3Encoder_h2(encodedY3_h1)

    id_start = (i - 1) * 9
    id_end = id_start + 9
    inp = Lambda(lambda x: x[:, id_start:id_end], output_shape=((9,)))(y4)
    encodedY4_h1 = Y4Encoder_h1(inp)
    EncodedY4[i] = Y4Encoder_h2(encodedY4_h1)


EncodedAnglesCat = Concatenate()(list(EncodedAngles.values()))
EncodedY2Cat = Concatenate()(list(EncodedY2.values()))
EncodedY3Cat = Concatenate()(list(EncodedY3.values()))
EncodedY4Cat = Concatenate()(list(EncodedY4.values()))

Encoded_CatAngles_h1 = Dense(100, activation="relu")(EncodedAnglesCat)
DeepAngles = Dense(50, activation="relu")(Encoded_CatAngles_h1)

Encoded_CatY2_h1 = Dense(100, activation="relu")(EncodedY2Cat)
DeepY2 = Dense(50, activation="relu")(Encoded_CatY2_h1)

Encoded_CatY3_h1 = Dense(100, activation="relu")(EncodedY3Cat)
DeepY3 = Dense(50, activation="relu")(Encoded_CatY3_h1)

Encoded_Cat4_h1 = Dense(100, activation="relu")(EncodedY4Cat)
DeepY4 = Dense(50, activation="relu")(Encoded_Cat4_h1)


FinalCat = Concatenate()([DeepFFT, DeepCoords, DeepAngles, DeepY2, DeepY3, DeepY4])

DeepFinal_h1 = Dense(200, activation='relu')(FinalCat)
DeepFinalOut = Dense(100, activation='relu')(DeepFinal_h1)

output_layer = Dense(2, activation='softmax')(DeepFinalOut)

model = Model(inputTensor, output_layer)

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function   categorical_crossentropy
                  optimizer="adam", # using the Adam optimiser adam
                  metrics=['accuracy']) # reporting the accuracy




model.fit(X_train, y_train, batch_size=2000, epochs=10,
              verbose=2,    validation_split=0.1)

model.fit(X_train, y_train, batch_size=2000, epochs=1,
              verbose=2,    validation_split=0.1)

model.fit(X_train[0:10000], y_train[0:10000], batch_size=2000, epochs=1,
              verbose=2,    validation_split=0.1)

model.evaluate(X_test, y_test, verbose=0)

model.save_weights('model_spherical_1_weights.h')
model.load_weights('model_spherical_1_weights.h')


# # Train on 14010796 samples, validate on 1556756 samples
# Epoch 1/10
# 5102s - loss: 0.2016 - acc: 0.9207 - val_loss: 0.1597 - val_acc: 0.9395
# Epoch 2/10
# 5017s - loss: 0.1455 - acc: 0.9453 - val_loss: 0.1158 - val_acc: 0.9571
# Epoch 3/10
# 5050s - loss: 0.1087 - acc: 0.9600 - val_loss: 0.1060 - val_acc: 0.9610
# Epoch 4/10
# 5041s - loss: 0.0979 - acc: 0.9643 - val_loss: 0.0936 - val_acc: 0.9661
# Epoch 5/10
# 5022s - loss: 0.0922 - acc: 0.9665 - val_loss: 0.0903 - val_acc: 0.9670
# Epoch 6/10
# 5016s - loss: 0.0886 - acc: 0.9679 - val_loss: 0.0903 - val_acc: 0.9675
# Epoch 7/10
# 5126s - loss: 0.0858 - acc: 0.9689 - val_loss: 0.0871 - val_acc: 0.9684
# Epoch 8/10
# 8662s - loss: 0.0835 - acc: 0.9698 - val_loss: 0.0820 - val_acc: 0.9704
# Epoch 9/10
# 5723s - loss: 0.0813 - acc: 0.9707 - val_loss: 0.0810 - val_acc: 0.9707
# Epoch 10/10
# 5527s - loss: 0.0791 - acc: 0.9715 - val_loss: 0.0782 - val_acc: 0.9718


## 1d fft of 1d hist
# Epoch 1/5
# 7232s - loss: 0.1416 - acc: 0.9452 - val_loss: 0.0990 - val_acc: 0.9634
# Epoch 2/5
# 6290s - loss: 0.0961 - acc: 0.9645 - val_loss: 0.0924 - val_acc: 0.9660
# Epoch 3/5
# 7287s - loss: 0.0869 - acc: 0.9681 - val_loss: 0.0898 - val_acc: 0.9668
# Epoch 4/5
# 7027s - loss: 0.0795 - acc: 0.9709 - val_loss: 0.0765 - val_acc: 0.9722
# Epoch 5/5
# 7232s - loss: 0.0743 - acc: 0.9729 - val_loss: 0.0699 - val_acc: 0.9747
# Epoch 1/5
# 5938s - loss: 0.0686 - acc: 0.9751 - val_loss: 0.0679 - val_acc: 0.9755
# Epoch 2/5
# 6018s - loss: 0.0649 - acc: 0.9766 - val_loss: 0.0635 - val_acc: 0.9772
# Epoch 3/5
# 5885s - loss: 0.0621 - acc: 0.9776 - val_loss: 0.0619 - val_acc: 0.9777
# Epoch 4/5
# 5896s - loss: 0.0597 - acc: 0.9785 - val_loss: 0.0628 - val_acc: 0.9773
# Epoch 5/5
# 5914s - loss: 0.0578 - acc: 0.9792 - val_loss: 0.0606 - val_acc: 0.9782


