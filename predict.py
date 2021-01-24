from predict_funcs_n_classes import *
import json


def main():
    in_arg = get_predict_input_args()
    
    model = load_checkpoint(in_arg.checkpoint)
        
    idx_to_class = {v: k for k, v in model.class_to_idx.items()} #convert indexes to class

    top_ps, top_classes = predict(os.getcwd()+'/'+in_arg.input,model, in_arg.top_k, in_arg.gpu)
    
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    
    #using cat_to_name to convert cats to the name of the flowers
    flower_names = []
    for class_number in top_classes.cpu().numpy().squeeze():
        flower_names.append(cat_to_name[idx_to_class[class_number]])
    
    
    top_ps = top_ps.tolist()[0]
    print("The flower names and probability of each class are:")
    
    for i,flower_name in enumerate(flower_names):
        print("Flower Name: ", flower_name,", probability: ", "{:.5f}".format(top_ps[i]))
    
if __name__ == "__main__":
    main()