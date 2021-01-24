
import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import torch.nn.functional as F
from workspace_utils import *


import argparse



#defining model_names for use in build_model() function:


def vgg16():
    return models.vgg16(pretrained=True)

def vgg13():
    return models.vgg13(pretrained=True)




model_call = {'vgg16': vgg16, 'vgg13': vgg13}


def get_input_args():
    Parse = argparse.ArgumentParser()
    Parse.add_argument('data_dir')
    Parse.add_argument('--save_dir', default="/")
    Parse.add_argument('--arch', default="vgg16")
    Parse.add_argument('--learning_rate',default="0.01", type=float)
    Parse.add_argument('--hidden_units',default="4096", type=int)
    Parse.add_argument('--epochs',default="20", type=int)
    Parse.add_argument('--gpu',action="store_true", default=False)


    args = Parse.parse_args()
   

    return args


def get_training_and_validation_data_loaders(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_and_validation_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=test_and_validation_transforms)


    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64)
    
    return trainloader, validationloader, train_data.class_to_idx
    
    
def build_model(model_name, hidden_units):
    model = model_call[model_name]()



# Freezing parameters so we don't backprop through them, this would have extensive computing overhead!
    for param in model.parameters():
        param.requires_grad = False

        #the pretrained model has model.classifier[0].in_features in_features in the first classifier layer, that is why I am defining as model.classifier[0].in_features as well.
    model.classifier = nn.Sequential(nn.Linear(model.classifier[0].in_features, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_units, int(hidden_units/2)),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(int(hidden_units/2), 102), #the final out_features is 102 because len(cat_to_name) is 102
                                     nn.LogSoftmax(dim=1))

    #-------------- end of building the model
    return model


def train_model(model, epochs, learning_rate, trainloader, validationloader, gpu):
    
    #defining the device to run the model training on a GPU if available, else use the CPU
    device=''
    if torch.cuda.is_available() & gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device);
    
    criterion = nn.NLLLoss() #the criteria method to check the loss when training the model

    # Only training the classifer parameters
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate) #I might test learning rates
 
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in keep_awake(range(epochs)):
        for inputs, labels in trainloader:

            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validationloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        validation_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(validationloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validationloader):.3f}")
                running_loss = 0
                model.train()
    model.optimizer = optimizer
    return model
    #-------------- end of training the model
    
def save_checkpoint(model, class_to_idx, architecture, hidden_units):
    model.class_to_idx = class_to_idx
    model.architecture = architecture
    model.hidden_units = hidden_units

    checkpoint = {'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'optimizer_state_dict': model.optimizer.state_dict(),
                  'hidden_units': model.hidden_units,
                  'architecture':  model.architecture}

    torch.save(checkpoint, 'checkpoint.pth')