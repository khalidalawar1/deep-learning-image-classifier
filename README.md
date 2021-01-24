# PyTorch Deep Learning Image Classifier

Code for an image classifier built with PyTorch, then convert it into a command line application

# Install

### To install, simply download the files `train.py` , `predict.py` , `funcs_n_classes.py` , and `predict_funcs_n_classes.py` and place them in the same folder.

# Usage

In general, the usage of this program consists of training the neural network model on classifications, and then supplying a new unseen image for it to predict and classify the image.

## The Project has 2 main files that you will call: `train.py` and `predict.py`

## Training the model
To train a model, run train.py with the path to the image folder and any other specified arguments:

### Example Command
```
python train.py flowers/ --arch vgg16 --hidden_units 4096 --epochs 5 --learning_rate 0.001 --gpu true --save_dir checkpoint.pth
```

### Input, Arguments, and Options
The program accommodates takes the following input and options:

```
usage: train.py [DATA_DIR] [--save_dir SAVE_DIR] [--arch ARCH]
                [--learning_rate LEARNING_RATE] [--hidden_units HIDDEN_UNITS]
                [--epochs EPOCHS] [--gpu GPU]
                
Provide image_dir, save_dir, architecture, hyperparameters such as
learningrate, num of hidden_units, epochs and whether to use gpu or not

arguments:
  DATA_DIR   path to image folder
  --save_dir SAVE_DIR   the place model checkpoints gets saved to, defaults to 'assets' directory
  --arch ARCH           currently, I implemented 2 options: vgg13 or vgg16
  --learning_rate LEARNING_RATE
                        learning_rate for model, from my experience smaller number like 0.001 reaches good accuracy, hence, defaults to 0.001
  --hidden_units HIDDEN_UNITS
                        hidden_units for model, defaults to 4096
  --epochs EPOCHS       epochs for model, defaults to 5
  --gpu GPU             enable or disable the gpu while learning. Note: If you train the model while gpu is enabled then you must predict with gpu enabled and vice versa.
```

### Training Data Set Format
It is important to structure the training data into a structure that the training program will digest. Within the folder you input in `DATA_DIR` in your `train.py` program call, make sure you have three sub folders: `train`, `valid` and `test`. Moreover, within each, have your images within subfolders named by a category index of these images. These would be used within the logic of the program for classification. 

## Predicting Images 
To predict an image into a classification, run predict.py using a saved model checkpoint and specify the image:
```
python predict.py --checkpoint checkpoint.pth --input unknown_flower.jpg
```

### Usage
```
usage: predict.py [INPUT] [--checkpoint CHECKPOINT]
                  [--top_k TOP_K] [--category_names CATEGORY_NAMES]
                  [--gpu GPU]

Provide input, checkpoint, top_k, category_names and gpu

arguments:
  --input INPUT         path to image to be predicted
  --checkpoint CHECKPOINT
                        path to checkpoint of the model used for predicting the classification of the input, defaults to 'checkpoint.pth'
  --top_k TOP_K         number of probable classifications of the input the program will show, defaults to 3
  --category_names CATEGORY_NAMES
                        path to file that maps indexes of trained categorires id's into english language categories readable by humans, defaults to 'cat_to_name.json'
  --gpu GPU             enable or disable the gpu while learning. Note: If the checkpoint model was trained while gpu is enabled then you must predict with gpu enabled and vice versa.
```


#### Note: most probably for your environment and especially those not running in the cloud, you might want to remove `workspace_utils.py` and all of its calls from the code
