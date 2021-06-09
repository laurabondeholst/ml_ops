## not finished

from model import MyAwesomeModel
import torch
from main import MODEL_PATH
import gzip
import os


def load_model(path =MODEL_PATH+"/checkpoint.pth"):
    # Input: path to model parameters
    # Output: returns model
    # Requirements: model parameters must fit the model from 'MyAwesomeModel'
    model = MyAwesomeModel()
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    return model

def load_data(path):
    # Input: path to raw data
    # Output: returns processed data

    with os.scandir(path) as dirs:
        for entry in dirs:
            p = path+'/'+str(entry.name)
            if entry.name[-2:] == 'gz':
                with gzip.open(p, 'rb') as f:
                    file_content = f.read()
            else: 
                file_content = open(p,'rb')


    print(file_content)

if __name__ == '__main__':
    load_data("C:/Users/Laura/Documents/MLOps/mlops_project/data/raw/")
    


