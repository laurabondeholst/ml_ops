from src.models.model import MyAwesomeModel
import numpy as np
import torch

def test_model_output():
    model = MyAwesomeModel()
    assert model.forward(torch.zeros((100, 1, 28, 28))).shape == torch.Size((100,10))




