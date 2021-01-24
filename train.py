
from funcs_n_classes import *



def main():
    in_arg = get_input_args()

    trainloader, validationloader, class_to_idx = get_training_and_validation_data_loaders(in_arg.data_dir)
    
   
    model = build_model(in_arg.arch, in_arg.hidden_units) #you can specify 2 different pre trained models: vgg13 or vgg16
    
    model = train_model(model,in_arg.epochs,in_arg.learning_rate, trainloader, validationloader, in_arg.gpu)
    
    save_checkpoint(model, class_to_idx, in_arg.arch, in_arg.hidden_units)

if __name__ == "__main__":
    main()