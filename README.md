This repository contains codes to train the shape-parameterized neural networks presented in the paper "A framework for physics-informed deep learning over freeform domains" by F. Mezzadri, J. Gasick and X. Qian. 
Links where the corresponding datasets can be downloaded are provided.

The codes have been run and tested on a system equipped with Python 3.6.13, Tensorflow 2.5.2, and Sciann 0.6.5.1

The codes are provided in the following directories:

1) The folder "rectangular" contains the file "rectangular.py", which trains and validates shape-parameterized neural networks for the rectangular beam problem as in Sec. 4.2 of the paper.
The subfolder "length" contains training data for a few lengths. Validation is performed by direct computation of the analytical solution in "rectangular.py"

2) The folder "circular" contains the file "circular.py", which trains shape-parameterized neural networks for the circular beam problem as in Sec. 4.2 of the paper.
The file "dataset_instructions.txt" contains a link to download the associated dataset (that can be used for training and validation).

3) The folder "platehole_5cp" contains the file "platehole_5cp.py", which trains a shape-parameterized PINN for the platehole problem with 5 control points (8 shape parameters) as in Sec. 4.3 - 4.4 of the paper.
The file "dataset_instructions.txt" contains a link to download the associated dataset (that can be used for training and validation).

4) The folder "platehole_11cp" contains the files "platehole.py" and "platehole_nn.py", which train a shape-parameterized PINN and NN, respectively, for the platehole problem with 11 control points (20 shape parameters) as in Sec. 4.6 of the paper.
The file "dataset_instructions.txt" contains a link to download the associated dataset (that can be used for training and validation) for the c=1, p=2 case.
