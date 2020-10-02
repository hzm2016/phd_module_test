All the four methods are written in one file CA2.py, but it quite clear;

run python CA2.py, "Load data", "plot_eigenvector", "image_reconstruction"
and train the 'LR' or 'PR' for handwritten classification will be implemented step by step;

You can change the 'algorithm' name from [LR', 'PR', 'LR_kernel'], the default is 'LR';
You can change the 'solver' of algorithm [None, 'gradient'], the default is None, which means calculated by normal equation.
You can change the 'mode' of algorithm [None, 'L1', 'L2'], the default is None, which means whether to add the regularization term.