# Student Name: Samuel Havard
# predict.py is an application used from the command line for the AIPND from Udacity
# The purpose of the application is to use a pre-trained network for image classifications
# Basic usage: 
#           Base minimum input: python predict.py input checkpointbasic usage
# options are:
#           Return top K most likely classes: python predict.py input checkpoint --top_k 3
#           Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
#           Use GPU for inference: python predict.py input checkpoint --gpu

import argparse
import json
import numpy as np
import torch

from PIL import Image
from torch.autograd import Variable

def main():
    args = get_arguments()
    cuda = args.cuda
    model = load_checkpoint(args.checkpoint, cuda)
    model.idx_to_class = dict([[v,k] for k, v in model.class_to_idx.items()])
    
    with open(args.categories, 'r') as f:
        cat_to_name = json.load(f)
      
    prob, classes = predict(args.input, model, args.cuda, topk=int(args.top_k))
    print([cat_to_name[x] for x in classes])
    
    
def get_arguments():
    """ 
    Retrieve command line keyword arguments
    """
    parser_msg = 'Predict.py takes 2 manditory command line arguments, \n\t1.The image to have a predition made and \n\t2. the checkpoint from the trained nerual network'
    parser = argparse.ArgumentParser(description = parser_msg)

    # Manditory arguments
    parser.add_argument("input", action="store")
    parser.add_argument("checkpoint", action="store")

    # Optional arguments
    parser.add_argument("--top_k", action="store", dest="top_k", default=5, help="Number of top results you want to view.")
    parser.add_argument("--category_names", action="store", dest="categories", default="cat_to_name.json", 
                        help="Number of top results you want to view.")
    parser.add_argument("--cuda", action="store_true", dest="cuda", default=False, help="Set Cuda True for using the GPU")

    return parser.parse_args()

        
def process_image(image):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    
    Parameters:
    image()
    
    Returns:
    
    '''
    image_ratio = image.size[1] / image.size[0]
    image = image.resize((256, int(image_ratio*256)))
    # left, upper, right, lower
#     image = image.crop((16, 16, 224, 224))
    half_the_width = image.size[0] / 2
    half_the_height = image.size[1] / 2
    image = image.crop((half_the_width - 112,
                       half_the_height - 112,
                       half_the_width + 112,
                       half_the_height + 112))
    
    image = np.array(image)
    image = image/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std_dev = np.array([0.229, 0.224, 0.225])
    
    image = (image - mean) / std_dev
    image = image.transpose((2, 0, 1))
    
    return torch.from_numpy(image)

    
def predict(image_path, model, cuda, topk):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    
    Parameters:
    image_path ():
    model ():
    cuda ():
    topk ():
    
    Returns:
    list of lists: a list of lists containing probabilities and classes
    '''
    if cuda:
        model.cuda()
    else:
        model.cpu()
  
    image = None
    model.eval()
    with Image.open(image_path) as img:
        image = process_image(img)
       
    image = Variable(image.unsqueeze(0), volatile=True)
    
    if cuda:
        image = image.cuda()

    output = model.forward(image.float())
    ps = torch.exp(output)
    prob, idx = ps.topk(topk)
    return [y for y in prob.data[0]], [model.idx_to_class[x] for x in idx.data[0]]


def load_checkpoint(filepath, cuda):
    ''' 
    loads a model, classifier, state_dict and class_to_idx from a torch save
    
    Parameters:
    filepath (str): string representation for the filepath to the checkpoint
    cuda (bool): boolean representation for availability of the GPU
    
    Returns:
    model: a rebuilt model from the checkpoint
    '''
    if cuda:
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
        
    return model

main()

