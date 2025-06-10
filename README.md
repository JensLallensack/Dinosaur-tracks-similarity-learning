This is the code for Lallensack, Falkingham, Orang, and Guimarães – "A similarity learning approach to explore the distribution of tridactyl dinosaurian trackmakers through space and time"

Data to run this code is stored at Figshare:
https://figshare.com/s/8db4d84ef99cb7de7619

For running the process_testset.py, we recommend to use Tensorflow v. 2.8.1 in a virtual environment such as conda to be compatible with TensorflowSimilarity.

Note that the provided "testset" is not a testset in the strict sense, as it partly contains images that some of the models were trained on.


This repository contains three folders (Obtain results, Synthetic track outline generation, Training):

1) Obtain results
This folder contains the final models (approaches 1 to 4) and code to run the models on unseen data (testset provided via figshare). The code will export UMAP plots as shown in the paper.

2) Synthetic track outline generation
This folder contains the scripts and data to generate the synthetic data. The .ods files containing the coordinates of the respective template tracks along with metadata are provided via figshare.

If "master.py" is run in Python, it will process all .ods, creating all classes, with a specified number of outlines per class (default: n=5).

3) Training
The code used to train the model (the "occurrences" dataset of dinosaur and bird tracks, as well as the synthetic dataset), along with the training and validation data, are provided via figshare.
