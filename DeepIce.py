from keras.layers import Dense, Input
from keras.layers import Concatenate
from keras.models import Model
from custom_nets import CoordinatesNet, AnglesNet, SphericalHarmonicsNet, FourierTransformTensor
from custom_nets import FourierTransformNet, ConcatNets, FinalLayer
import numpy as np

class DeepIce():
    def __init__(self, n_neighbours, Nets=None, NetsParams=None, num_classes=2, optimizer="adam",
                 loss='categorical_crossentropy', metrics=['accuracy'],
                 FourierBins=30, FourierRange=[-6.0, 6.0]):

        self.n_neighbours = n_neighbours
        self.len_input = self.n_neighbours*3
        self.num_classes = num_classes
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

        # Which subnetworks to build. Nets["xx"] is bool
        if Nets == None:
            self.CartCoordNet = True
            self.FourierNet = True
            self.SpherCoordNet = True
            self.SpherHarmNet = True
        else:
            self.CartCoordNet = Nets["CartCoord"]
            self.FourierNet = Nets["Fourier"]
            self.SpherCoordNet = Nets["SpherCoord"]
            self.SpherHarmNet = Nets["SpherHarm"]

        # Dict with subnetworks params (number of layes and neurons)
        self.NetParams = {}

        # Default architecture values:
        nets = ["CartCoord", "Fourier", "SpherCoord", "SpherHarm"]
        nets_default_params = {}
        nets_default_params["CartCoord"] = [[250, 50], [250, 50]]
        nets_default_params["SpherCoord"] = [[250, 50], [250, 50]]
        nets_default_params["SpherHarm"] = [[250, 50], [250, 50], 200, 50]
        nets_default_params["Fourier"] = [[250, 50], [250, 50]]
        nets_default_params["Final"] = [250, 50, 5]
        if NetsParams == None:
            NetsParams = {}

        for net in nets:
            self.NetParams[net] = {}
            # set to default values:
            self.NetParams[net]["encoder_dims"] = nets_default_params[net][0]
            self.NetParams[net]["deepconcat_dims"] = nets_default_params[net][1]

            # Update any values that were user set:
            if net in NetsParams.keys():
                if "encoder_dims" in NetsParams[net].keys():
                    self.NetParams[net]["encoder_dims"] = NetsParams[net]["encoder_dims"]
                if "deepconcat_dims" in NetsParams[net].keys():
                    self.NetParams[net]["deepconcat_dims"] = NetsParams[net]["deepconcat_dims"]

        #Fourier Bin Parameters:
        self.NetParams["Fourier"]["nbins"] = FourierBins
        self.NetParams["Fourier"]["ValueRange"] = FourierRange

        # Additional Spher Harm Cat Layers:
        if "SpherHarm" in NetsParams.keys():
            self.NetParams["SpherHarm"]["Cat1"] = NetsParams["SpherHarm"]["cat1"]
            self.NetParams["SpherHarm"]["Cat2"] = NetsParams["SpherHarm"]["cat2"]
        else:
            self.NetParams["SpherHarm"]["Cat1"] = nets_default_params["SpherHarm"][2]
            self.NetParams["SpherHarm"]["Cat2"] = nets_default_params["SpherHarm"][3]


        # Final layer architecture:
        if "Final" in NetsParams.keys():
            self.NetParams["Final"] = NetsParams["Final"]
        else:
            self.NetParams["Final"] = nets_default_params["Final"]


    def BuildModel(self):

        # Input tensor order is x1, y1, z1, x2, y2,z2, ..., x10, y10, z10
        # where number corresponds to nearest neighbour ID
        self.inputTensor = Input(shape=(self.len_input,), name='InputCoordinate')

        # Tensors to pass into final layer:
        self.outputTensors = []

        # Fourier Transform Network
        if self.FourierNet == True:
            print('Building Fourier Transform Network...')
            nbins = self.NetParams["Fourier"]["nbins"]
            ValueRange = self.NetParams["Fourier"]["ValueRange"]
            encoder_dims = self.NetParams["Fourier"]["encoder_dims"]
            deepconcat_dims = self.NetParams["Fourier"]["deepconcat_dims"]
            print('... encoding network dimensions {}'.format(encoder_dims))
            print('... combined network dimensions {}'.format(deepconcat_dims))
            self.fft_x, self.fft_y, self.fft_z = FourierTransformTensor(self.inputTensor, self.n_neighbours,
                                                                        nbins=nbins, ValueRange=ValueRange)
            self.DeepFFT = FourierTransformNet(self.fft_x, self.fft_y, self.fft_z, encoder_dims=encoder_dims,
                                               deepconcat_dims=deepconcat_dims)
            self.outputTensors.append(self.DeepFFT)


        # Cartesian Coordinates Network
        if self.CartCoordNet == True:
            print('Building Cartesian Coordinates Network...')
            encoder_dims = self.NetParams["CartCoord"]["encoder_dims"]
            deepconcat_dims = self.NetParams["CartCoord"]["deepconcat_dims"]
            print('... encoding network dimensions {}'.format(encoder_dims))
            print('... combined network dimensions {}'.format(deepconcat_dims))
            self.DeepCoords = CoordinatesNet(self.inputTensor, self.n_neighbours,
                                             encoder_dims=encoder_dims,
                                             deepconcat_dims=deepconcat_dims)
            self.outputTensors.append(self.DeepCoords)


        # Spherical Coordinates Network
        if self.SpherCoordNet == True:
            print('Building Spherical Coordinates Network...')
            encoder_dims = self.NetParams["SpherCoord"]["encoder_dims"]
            deepconcat_dims = self.NetParams["SpherCoord"]["deepconcat_dims"]
            print('... encoding network dimensions {}'.format(encoder_dims))
            print('... combined network dimensions {}'.format(deepconcat_dims))
            self.DeepAngles = AnglesNet(self.inputTensor, self.len_input,
                                        self.n_neighbours, encoder_dims=encoder_dims,
                                        deepconcat_dims=deepconcat_dims)
            self.outputTensors.append(self.DeepAngles)



        # Spherical Harmonics Network
        if self.SpherHarmNet == True:
            print('Building Spherical Harmonics Network...')
            encoder_dims = self.NetParams["SpherHarm"]["encoder_dims"]
            deepconcat_dims = self.NetParams["SpherHarm"]["deepconcat_dims"]
            print('... encoding network dimensions {}'.format(encoder_dims))
            print('... combined network dimensions {}'.format(deepconcat_dims))
            cat1 = self.NetParams["SpherHarm"]["Cat1"]
            cat2 = self.NetParams["SpherHarm"]["Cat2"]

            self.DeepY2 = SphericalHarmonicsNet(self.inputTensor, self.len_input, self.n_neighbours,
                                           spherical_ID=2, encoder_dims=encoder_dims,
                                           deepconcat_dims=deepconcat_dims)
            self.DeepY3 = SphericalHarmonicsNet(self.inputTensor, self.len_input, self.n_neighbours,
                                           spherical_ID=3, encoder_dims=encoder_dims,
                                           deepconcat_dims=deepconcat_dims)
            self.DeepY4 = SphericalHarmonicsNet(self.inputTensor, self.len_input, self.n_neighbours,
                                           spherical_ID=4, encoder_dims=encoder_dims,
                                           deepconcat_dims=deepconcat_dims)

            self.YCat = Concatenate(name='SH_Concat')([self.DeepY2, self.DeepY3, self.DeepY4])
            self.DeepYCat_h1 = Dense(cat1, activation='relu', name='SH_h1')(self.YCat)
            self.DeepY = Dense(cat2, activation='relu', name='SH_h2')(self.DeepYCat_h1)
            self.outputTensors.append(self.DeepY)

        if len(self.outputTensors) == 1:
            self.output_layer = FinalLayer(self.outputTensors[0], skip=True, num_classes=self.num_classes)
        else:
            hidden_layers = self.NetParams["Final"]
            FinalCat = ConcatNets(*self.outputTensors)
            self.output_layer = FinalLayer(FinalCat, hidden_layers=hidden_layers, skip=False, num_classes=self.num_classes)

    def CompileModel(self):
        self.model = Model(self.inputTensor, self.output_layer)

        # default is Adam optimiser and categorical_crossentropy
        self.model.compile(loss=self.loss,
                      optimizer=self.optimizer,  # using the Adam optimiser adam
                      metrics=self.metrics)  # reporting the accuracy

    def ImportWeights(self, filename):
        self.model.load_weights(filename)

    def ExportWeights(self, filename):
        self.model.save_weights(filename)

    def get_data(self, filename, test=False, train=True):
        data = np.load(filename)  #
        if test:
            X, y = data['X_test'], data['y_test']
        elif train:
            X, y = data['X_train'], data['y_train']

        cols_want = self.n_neighbours * 3
        X = X[:, 0:cols_want]
        return X, y

    def TrainModel(self, X, y, batch_size=2000, epochs=1, verbose=1, validation_split=0.1, shuffle=True):
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=verbose,
                      validation_split=validation_split, shuffle=shuffle)


    def Predict(self, data_file, outputfile, num_mols):

        data = np.load(data_file)
        X = data['input'][:, 0:self.n_neighbours * 3]

        y_pred = self.model.predict(X)
        y_pred = np.argmax(y_pred, axis=-1)
        print(type(num_mols))
        num_mols = int(num_mols)
        print(type(num_mols))        
        y_pred = np.reshape(y_pred, (-1, num_mols))
        np.savetxt(outputfile, y_pred, delimiter=",", fmt='%i')


    def EvaluateModel(self, X, y, verbose=1):
        print(self.model.evaluate(X, y, verbose=verbose))
