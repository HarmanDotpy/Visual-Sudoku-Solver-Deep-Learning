# Visual Sudoku Solver 

In this repo, a sudoku solver is designed to solve directly from images (Visual Sudoku). Recurrent Relational Networks are used for the task of solving the sudoku.

## Task

We need to make a model which uses takes in an input sudoku board made of handwritten digits and outputs the the solved sudoku board in symbolic form (in the form of digits on a computer).
For training purposes we are given pairs of visual sudoku boards as follows

![0](https://user-images.githubusercontent.com/50492433/120084470-009d7280-c0ee-11eb-96a3-dbbf2ba3cd51.png) (unsolved) ![0 (1)](https://user-images.githubusercontent.com/50492433/120084473-02ffcc80-c0ee-11eb-9211-09192eb955ea.png) (solved)

(Note that these are 8\*8 sudoku boards where each column, each row and each bloack of size 2\*4 (long side along x axis) is filled with 8 unique digits in the solution)

For converting individual handwritten cells to symbolic data, we can use all the training images, extract all sub-images from it which (8\*8\=64 images from one sudoku board image) and then do clustering/classification techniques. For (semi) supervision, we can use 1 labeled image from each class which is given separately. We use a combined technique of kmeans+Unsupervised Data augmentation([UDA](https://arxiv.org/abs/1904.12848)) to make a classifier which gives 95%+ accuracy using just these 9 labeled images and a larger set of unlabeled images

labelled image (1 per class, class 0 to 8 from left to right)

![Screenshot 2021-05-30 at 2 29 05 AM](https://user-images.githubusercontent.com/50492433/120084551-d13b3580-c0ee-11eb-823b-69c899e5ae47.png)


## Running the solver

### Without Joint training
Performing Unsupervised clustering (using [UDA](https://arxiv.org/abs/1904.12848)) then using the classifier made in the UDA step to convert visual sudoku boards into symbolic boards (will have some noise)  and then training the RRN on these input-output symbolic sudoku boards.
Noise in labels limits the ability of the RRN to learn the rules of sudoku

```bash

run_solver.sh <path_to_train> <path_to_test_query> <path_to_sample_imgs> <path_to_out_csv>

```

### With Joint training
Similar to the earlier part but this time, the classifier that we get from UDA is fine tuned while training the RRN. ie The pretrained classifier and RRN are trained jointly so that both improve each other

```bash

run_solver.sh <path_to_train> <path_to_test_query> <path_to_sample_imgs> <path_to_out_csv> true

```

In the above commands
- <path_to_train> directory has to sub directories, <path_to_train>/query/ and <path_to_train>/target/. Both these subdirectories have images of sudoku boards made of handwritten digits. Solution of the board <path_to_train>/query/n.png should be <path_to_train>/target/n.png where n is the number of the board (eg 0.png, 1.png ......)
- <path_to_test_query> has unsolved visual boards just like in <path_to_train>/query/ that will be solved after model is trained (for testing purposes)
- <path_to_sample_imgs> is a numpy file (.npy) of shape (10,784) having one labelled image of each class (digit)
- <path_to_out_csv> is where the result of solving the unsolved sudoku boards present in <path_to_test_query> will be stored in symbolic form (in the form of digits).





# Rough notes
## Recurrent Relational Network

- [x] RRN (as per paper with slight modification) ([reference](https://github.com/wDaniec/pytorch-RNN))

## Joint Training for solving visual sudoku

1. [SatNet](https://arxiv.org/pdf/1905.12149.pdf) : idea of training classifier and rrn together (but this was possible as they had symbolic ground truth sudoku output tables)
2. Optimize loss function of RRN with two more loss functions on (same or tied weights) classifier as penalty terms 
3. we may use here cGAN with batchnorm or bit more "richer" architecture (wrt paper) .. ? (RRN is already modified)
