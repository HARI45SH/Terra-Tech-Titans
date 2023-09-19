import torch
import torch.nn as nn
from python_files.model import Mobilenet_reg

class Load_model():
    def __init__(self,Path):
        self.model=Mobilenet_reg()
        self.PATH=Path
        if self.PATH!=None:
            self.model.load_state_dict(self.PATH)
    
    def __call__(self,image):
        class_pred,slip_rough=self.model(image)
        slip,rough=slip_rough[0][0].item(),slip_rough[0][1].item()

        return class_pred.argmax(dim=1).item(),slip,rough