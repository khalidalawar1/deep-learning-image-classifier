# PyTorch Deep Learning Image Classifier

Code for an image classifier built with PyTorch, then convert it into a command line application

# Install

### To install, simply download the files `train.py` , `predict.py` , `funcs_n_classes.py` , and `predict_funcs_n_classes.py` and place them in the same folder.

# Usage

In general, the usage of this program consists of training the neural network model on classifications, and then supplying a new unseen image for it to predict and classify the image.

## The Project has 2 main files that you will call: `train.py` and `predict.py`

## Training the model

### Example Command
```
python train.py flowers/x` --arch vgg16 --hidden_units 4096 --epochs 5 --learning_rate 0.001 --gpu true --save_dir checkpoint.pth
```

### Input, Arguments, and Options
The program accommodates takes the following input and options:

```
usage: train.py [-h] [DATA_DIR] [--save_dir SAVE_DIR] [--arch ARCH]
                [--learning_rate LEARNING_RATE] [--hidden_units HIDDEN_UNITS]
                [--epochs EPOCHS] [--gpu GPU]
                
Provide image_dir, save_dir, architecture, hyperparameters such as
learningrate, num of hidden_units, epochs and whether to use gpu or not

optional arguments:
  DATA_DIR   path to image folder
  --save_dir SAVE_DIR   the place model checkpoints gets saved to, defaults to 'assets' directory
  --arch ARCH           2 options: vgg13 or vgg16
  --learning_rate LEARNING_RATE
                        learning_rate for model, from my experience smaller number like 0.001 reaches good accuracy, hence, defaults to 0.001
  --hidden_units HIDDEN_UNITS
                        hidden_units for model, defaults to 4096
  --epochs EPOCHS       epochs for model, defaults to 5
  --gpu GPU             enable or disable the gpu while learning. Note: If you train the model while gpu is enabled then you must predict with gpu enabled and vice versa.
```

### 
