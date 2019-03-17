# DeepIce

![Alt text](deepIce_v2.png?raw=true "DeepIce Diagram")


## Deep Neural Networks for identifying the phase of molecules

```
@article{deepice,
   author = {Fulford, Maxwell and Salvalaglio, Matteo and Molteni, Carla},
   title = {DeepIce: a Deep Neural Network Approach to Identify Ice and Water Molecules},
   journal = {Journal of Chemical Information and Modeling},
   doi = {10.1021/acs.jcim.9b00005}}
```


4 main networks: 
 - Spherical Harmonics Network
 - Fourier Transform Network
 - Catersian Coordinates Network
 - Spherical Coordinates Network 
 
 ### Usage:
 
 ```
 python main_deepice.py --help
```

Training with 10 nearest neighbours, batch size of 30 and 5 epochs:

```
python main_deepIce.py --Train --nearest_neighbours 10 --batch_size 30 --n_epochs 5 --weights_file 'models/deepice_nn10.h5' --data 'data/deepice_traindata.npz' --output_weights 'models/deepice_nn10_trained.h5
```

Predicting on a simulation slab with 5760 molecules

```
python main_deepIce.py --Predict --data_file 'simulation_data.npz' --nearest_neighbours 10 --num_mols 5760
```

Evaluating accuracy on data set:

```
python main_deepIce.py --Evaluate --data_file 'data/deepice_testdata.npz' --nearest_neighbours 13
```

##

### Classification error DeepIce compared existing approaches:
![Alt text](deepIce_logError.png?raw=true "DeepIce Diagram")
