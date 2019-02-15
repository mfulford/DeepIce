from DeepIce import DeepIce
from optparse import OptionParser


def main():
    parser = OptionParser()

    parser.add_option('--Train', action="store_true", default=False,
                      help="Would you like to train the model?")

    parser.add_option("--Evaluate", action="store_true", default=False,
                      help="Would you like to train the model?")

    parser.add_option("--Predict", action="store_true", default=False,
                      help="Would you like to predict on a simulation?")

    parser.add_option("--nn", "--nearest_neighbours", type="int", default=10, dest="n_neighbours",
                      help='How many nearest neighbours (int)?')

    parser.add_option("--bS", "--batch_size", type="int", default=2000, dest="batch_size",
                      help='Batch Size for training (int).')

    parser.add_option("--n_epochs", type="int", default=1, dest="n_epochs",
                      help='How many epochs (int)?')

    parser.add_option('--W', '--weights_file',  dest="weights_file", default=None)

    parser.add_option('--data', '--data_file', help='Train, evaluate or prediction nearest neighbour data file', dest="data_file")

    parser.add_option('--oW', '--output_weights_file', dest="output_weights_file", default=None)

    parser.add_option('--num_mols', dest="num_mols", help=" Number of molecules in simulation (for prediction mode only)", default=5760)

    parser.add_option('--pO', '--prediction_output_file', dest="prediction_output_file", default=None)

    (options, args) = parser.parse_args()

    n_epochs = options.n_epochs
    n_neighbours = options.n_neighbours
    batch_size = options.batch_size
    weights_file = options.weights_file
    prediction_output_file = options.prediction_output_file
    data_file = options.data_file
    output_weights_file = options.output_weights_file
    num_mols = options.num_mols

    if data_file == None:
        print('Error!. Need a data file (--data_file is None)...')
        print('Exiting program')
        return



    print('DeepIce with {} nearest neighbours'.format(n_neighbours))
    print('DeepIce with Batch Size: {}'.format(batch_size))


    if weights_file != None:
        print('Load Weights File: {}'.format(weights_file))

    if output_weights_file == None:
        output_weights_file = 'deepice_trained_weights_{}nn_{}epochs.h'.format(n_neighbours, n_epochs)

    print('Data file: {}'.format(data_file))



    deepice = DeepIce(n_neighbours=n_neighbours, Nets=None, NetsParams=None, num_classes=2, optimizer="adam",
                     loss='categorical_crossentropy', metrics=['accuracy'],
                      FourierBins=30, FourierRange=[-6.0, 6.0])

    deepice.BuildModel()
    deepice.CompileModel()

    if weights_file != None:
        deepice.ImportWeights(filename=weights_file) #'models/deepice_11nn_20epochs.h'

    # "data/input_rotated_1D_3loops_TrainTest_14nn.npz"
    if options.Train == True:
        X, y = deepice.get_data(filename = data_file, test=False, train=True)
        print('Training for {} epochs'.format(n_epochs))
        deepice.TrainModel(X, y, batch_size=batch_size, epochs=n_epochs, verbose=1, validation_split=0.1, shuffle=True)
        print('Saving weights to... '.format(output_weights_file))
        deepice.ExportWeights(filename=output_weights_file)

    if options.Evaluate == True:
        X, y = deepice.get_data(filename = data_file, test=True, train=False)
        deepice.EvaluateModel(X, y, verbose=1)

    if options.Predict == True:
        if prediction_output_file == None:
            prediction_output_file = 'prediction_results_{}mols.dat'.format(num_mols)
        print('Number of molecules in simulation: {}'.format(num_mols))
        print('Prediction Output File: {}'.format(prediction_output_file))

        deepice.Predict(data_file=data_file, outputfile=prediction_output_file, num_mols=num_mols)



if __name__== "__main__":
  main()
