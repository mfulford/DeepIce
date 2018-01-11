from keras.layers import Dense
from keras.layers import Concatenate, Lambda
from keras import backend as K

from histogram_layer import Layer_Hist
from angle_layer import Layer_Angles
from spherical_layer import Layer_Y
from fft_layer import Layer_FFT


def CoordinatesNet(inputTensor, n_neighbours, encoder_dims=[250, 250, 50], deepconcat_dims=[250, 50]):
    n_encoder_layers = len(encoder_dims)
    n_deepconcat_layers = len(deepconcat_dims)

    CoordinateEncoder_1 = Dense(encoder_dims[0], activation="relu")
    if n_encoder_layers == 2:
        CoordinateEncoder_2 = Dense(encoder_dims[1], activation="relu")
    elif n_encoder_layers == 3:
        CoordinateEncoder_2 = Dense(encoder_dims[1], activation="relu")
        CoordinateEncoder_3 = Dense(encoder_dims[2], activation="relu")

    EncodedCoords = {}
    for i in range(1,n_neighbours+1):
        id_start = (i-1)*3
        id_end = id_start + 3
        xyz = Lambda(lambda x: x[:, id_start:id_end], output_shape=((3,)))(inputTensor)

        if n_encoder_layers == 1:
            EncodedCoords[i] = CoordinateEncoder_1(xyz)
        elif n_encoder_layers == 2:
            encoded_h1 = CoordinateEncoder_1(xyz)
            EncodedCoords[i] = CoordinateEncoder_2(encoded_h1)
        elif n_encoder_layers == 3:
            encoded_h1 = CoordinateEncoder_1(xyz)
            encoded_h2 = CoordinateEncoder_2(encoded_h1)
            EncodedCoords[i] = CoordinateEncoder_3(encoded_h2)

    EncodedCoordsCat = Concatenate()(list(EncodedCoords.values()))

    if n_deepconcat_layers == 1:
        DeepCoords = Dense(deepconcat_dims[0], activation="relu")(EncodedCoordsCat)
    elif n_deepconcat_layers == 2:
        Encoded_CatCoords_1 = Dense(deepconcat_dims[0], activation="relu")(EncodedCoordsCat)  # 500
        DeepCoords = Dense(deepconcat_dims[1], activation="relu")(Encoded_CatCoords_1)
    elif n_deepconcat_layers == 3:
        Encoded_CatCoords_1 = Dense(deepconcat_dims[0], activation="relu")(EncodedCoordsCat) #500
        Encoded_CatCoords_2 = Dense(deepconcat_dims[1], activation="relu")(Encoded_CatCoords_1) #500
        DeepCoords = Dense(deepconcat_dims[2], activation="relu")(Encoded_CatCoords_2)

    return DeepCoords

def FourierTransformTensor(inputTensor, n_neighbours, nbins=30, ValueRange=[-6.0, 6.0]):

    # indices of x, y and z coordinates within inputTensor
    id_x = K.tf.constant([(x-1)*3 for x in range(1,n_neighbours+1)])
    id_y = K.tf.constant([(x-1)*3 +1 for x in range(1,n_neighbours+1)])
    id_z = K.tf.constant([(x-1)*3 +2 for x in range(1,n_neighbours+1)])

    # x, y, z coordinates as 3 individual tensors
    x_tensor = Lambda(lambda x, indices: K.tf.gather(x, indices, axis=1), output_shape=((n_neighbours,)), arguments={'indices': id_x})(inputTensor)
    y_tensor = Lambda(lambda x, indices: K.tf.gather(x, indices, axis=1), output_shape=((n_neighbours,)), arguments={'indices': id_y})(inputTensor)
    z_tensor = Lambda(lambda x, indices: K.tf.gather(x, indices, axis=1), output_shape=((n_neighbours,)), arguments={'indices': id_z})(inputTensor)

    # 1D histograms of coordinates
    hist_x = Layer_Hist(nbins=nbins, ValueRange=ValueRange, input_shape=(n_neighbours,))(x_tensor)
    hist_y = Layer_Hist(nbins=nbins, ValueRange=ValueRange, input_shape=(n_neighbours,))(y_tensor)
    hist_z = Layer_Hist(nbins=nbins, ValueRange=ValueRange, input_shape=(n_neighbours,))(z_tensor)

    # 1D Fourier Transform of 1D histograms
    fft_x = Layer_FFT(fftDim=1, input_shape=(nbins,))(hist_x)
    fft_y = Layer_FFT(fftDim=1, input_shape=(nbins,))(hist_y)
    fft_z = Layer_FFT(fftDim=1, input_shape=(nbins,))(hist_z)

    return fft_x, fft_y, fft_z

def FourierTransformNet(fft_x, fft_y, fft_z, encoder_dims=[100, 20], deepconcat_dims=[100, 50]):
    n_encoder_layers = len(encoder_dims)
    n_deepconcat_layers = len(deepconcat_dims)

    FFT_Encoder_1D_1 = Dense(encoder_dims[0], activation="relu")
    if n_encoder_layers == 2:
        FFT_Encoder_1D_2 = Dense(encoder_dims[1], activation="relu")
    elif n_encoder_layers == 3:
        FFT_Encoder_1D_2 = Dense(encoder_dims[1], activation="relu")
        FFT_Encoder_1D_3 = Dense(encoder_dims[2], activation="relu")


    if n_encoder_layers == 1:
        fft_1d_x_out = FFT_Encoder_1D_1(fft_x)
        fft_1d_y_out = FFT_Encoder_1D_1(fft_y)
        fft_1d_z_out = FFT_Encoder_1D_1(fft_z)

    elif n_encoder_layers == 2:
        fft_1d_x_1 = FFT_Encoder_1D_1(fft_x)
        fft_1d_y_1 = FFT_Encoder_1D_1(fft_y)
        fft_1d_z_1 = FFT_Encoder_1D_1(fft_z)

        fft_1d_x_out = FFT_Encoder_1D_2(fft_1d_x_1)
        fft_1d_y_out = FFT_Encoder_1D_2(fft_1d_y_1)
        fft_1d_z_out = FFT_Encoder_1D_2(fft_1d_z_1)

    elif n_encoder_layers == 3:
        fft_1d_x_1 = FFT_Encoder_1D_1(fft_x)
        fft_1d_y_1 = FFT_Encoder_1D_1(fft_y)
        fft_1d_z_1 = FFT_Encoder_1D_1(fft_z)

        fft_1d_x_2 = FFT_Encoder_1D_2(fft_1d_x_1)
        fft_1d_y_2 = FFT_Encoder_1D_2(fft_1d_y_1)
        fft_1d_z_2 = FFT_Encoder_1D_2(fft_1d_z_1)

        fft_1d_x_out = FFT_Encoder_1D_3(fft_1d_x_2)
        fft_1d_y_out = FFT_Encoder_1D_3(fft_1d_y_2)
        fft_1d_z_out = FFT_Encoder_1D_3(fft_1d_z_2)

    Cat_fft_1d = Concatenate()([fft_1d_x_out, fft_1d_y_out, fft_1d_z_out])

    if n_deepconcat_layers == 1:
        DeepFFT = Dense(deepconcat_dims[0], activation="relu")(Cat_fft_1d)
    elif n_deepconcat_layers == 2:
        Encoded_CatFFT_1d_1 = Dense(deepconcat_dims[0], activation="relu")(Cat_fft_1d)
        DeepFFT = Dense(deepconcat_dims[1], activation="relu")(Encoded_CatFFT_1d_1)
    elif n_deepconcat_layers == 3:
        Encoded_CatFFT_1d_1 = Dense(deepconcat_dims[0], activation="relu")(Cat_fft_1d)
        Encoded_CatFFT_1d_2 = Dense(deepconcat_dims[1], activation="relu")(Encoded_CatFFT_1d_1)
        DeepFFT = Dense(deepconcat_dims[2], activation="relu")(Encoded_CatFFT_1d_2)

    return DeepFFT

def AnglesNet(inputTensor, len_input, n_neighbours, encoder_dims=[250, 250, 50], deepconcat_dims=[100, 50]):
    n_encoder_layers = len(encoder_dims)
    n_deepconcat_layers = len(deepconcat_dims)

    angles = Layer_Angles(input_shape=(len_input,))(inputTensor)

    AnglesEncoder_h1 = Dense(encoder_dims[0], activation="relu")
    if n_encoder_layers == 2:
        AnglesEncoder_h2 = Dense(encoder_dims[1], activation="relu")
    elif n_encoder_layers == 3:
        AnglesEncoder_h2 = Dense(encoder_dims[1], activation="relu")
        AnglesEncoder_h3 = Dense(encoder_dims[2], activation="relu")



    EncodedAngles = {}
    for i in range(1, n_neighbours+1):
        id_start = (i - 1) * 3
        id_end = id_start + 3
        inp = Lambda(lambda x: x[:, id_start:id_end], output_shape=((3,)))(angles)
        if n_encoder_layers == 1:
            EncodedAngles[i] = AnglesEncoder_h1(inp)
        elif n_encoder_layers == 2:
            encodedAngles_h1 = AnglesEncoder_h1(inp)
            EncodedAngles[i] = AnglesEncoder_h2(encodedAngles_h1)
        elif n_encoder_layers == 3:
            encodedAngles_h1 = AnglesEncoder_h1(inp)
            encodedAngles_h2 = AnglesEncoder_h2(encodedAngles_h1)
            EncodedAngles[i] = AnglesEncoder_h3(encodedAngles_h2)

    EncodedAnglesCat = Concatenate()(list(EncodedAngles.values()))

    if n_deepconcat_layers == 1:
        DeepAngles = Dense(deepconcat_dims[0], activation="relu")(EncodedAnglesCat)
    elif n_deepconcat_layers == 2:
        Encoded_CatAngles_h1 = Dense(deepconcat_dims[0], activation="relu")(EncodedAnglesCat)  # 500
        DeepAngles = Dense(deepconcat_dims[1], activation="relu")(Encoded_CatAngles_h1)
    elif n_deepconcat_layers == 3:
        Encoded_CatAngles_h1 = Dense(deepconcat_dims[0], activation="relu")(EncodedAnglesCat) #500
        Encoded_CatCoords_2 = Dense(deepconcat_dims[1], activation="relu")(Encoded_CatAngles_h1) #500
        DeepAngles = Dense(deepconcat_dims[2], activation="relu")(Encoded_CatCoords_2)

    return DeepAngles

def SphericalHarmonicsNet(inputTensor, len_input, n_neighbours, spherical_ID=2, encoder_dims=[100, 10], deepconcat_dims=[100, 50]):
    n_encoder_layers = len(encoder_dims)
    n_deepconcat_layers = len(deepconcat_dims)
    nl = (spherical_ID *2) + 1 #2l + 1

    y = Layer_Y(spherical_ID=spherical_ID, input_shape=(len_input,))(inputTensor)

    YEncoder_h1 = Dense(encoder_dims[0], activation="relu")
    if n_encoder_layers == 2:
        YEncoder_h2 = Dense(encoder_dims[1], activation="relu")
    elif n_encoder_layers == 3:
        YEncoder_h2 = Dense(encoder_dims[1], activation="relu")
        YEncoder_h3 = Dense(encoder_dims[2], activation="relu")


    EncodedY = {}
    for i in range(1, n_neighbours+1):
        id_start = (i - 1) * nl
        id_end = id_start + nl
        inp = Lambda(lambda x: x[:, id_start:id_end], output_shape=((nl,)))(y)
        if n_encoder_layers == 1:
            EncodedY[i] = YEncoder_h1(inp)
        elif n_encoder_layers == 2:
            encodedY_h1 = YEncoder_h1(inp)
            EncodedY[i] = YEncoder_h2(encodedY_h1)
        elif n_encoder_layers == 3:
            encodedY_h1 = YEncoder_h1(inp)
            encodedY_h2 = YEncoder_h2(encodedY_h1)
            EncodedY[i] = YEncoder_h3(encodedY_h2)


    EncodedYCat = Concatenate()(list(EncodedY.values()))

    if n_deepconcat_layers == 1:
        DeepY = Dense(deepconcat_dims[0], activation="relu")(EncodedYCat)
    elif n_deepconcat_layers == 2:
        Encoded_CatY_h1 = Dense(deepconcat_dims[0], activation="relu")(EncodedYCat)
        DeepY = Dense(deepconcat_dims[1], activation="relu")(Encoded_CatY_h1)
    elif n_deepconcat_layers == 3:
        Encoded_CatY_h1 = Dense(deepconcat_dims[0], activation="relu")(EncodedYCat)
        Encoded_CatY_h2 = Dense(deepconcat_dims[1], activation="relu")(Encoded_CatY_h1)
        DeepY = Dense(deepconcat_dims[2], activation="relu")(Encoded_CatY_h2)

    return DeepY

def ConcatNets(*args):
    if len(args) == 0:
        print("Error! Need at least one network to train the model")
        return 0
    elif len(args) == 1:
        return args

    nets = {}
    for idx, arg in enumerate(args):
        nets[idx] = arg

    FinalCat = Concatenate()(list(nets.values()))
    return FinalCat

def FinalLayer(FinalCat, hidden_layers = [200, 100], skip=False, num_classes=2):
    if skip == True:
        output_layer = Dense(num_classes, activation='softmax')(FinalCat)
    else:
        n_layers = len(hidden_layers)
        if n_layers == 1:
            DeepFinalOut = Dense(hidden_layers[0], activation='relu')(FinalCat)
        elif n_layers == 2:
            DeepFinal_h1 = Dense(hidden_layers[0], activation='relu')(FinalCat)
            DeepFinalOut = Dense(hidden_layers[1], activation='relu')(DeepFinal_h1)
        elif n_layers == 3:
            DeepFinal_h1 = Dense(hidden_layers[0], activation='relu')(FinalCat)
            DeepFinal_h2 = Dense(hidden_layers[1], activation='relu')(DeepFinal_h1)
            DeepFinalOut = Dense(hidden_layers[2], activation='relu')(DeepFinal_h2)

        output_layer = Dense(num_classes, activation='softmax')(DeepFinalOut)
    return output_layer
