from python_files.model_eval import Load_model
import torch
m=Load_model(Path=None)
print(m(
    torch.randn(1,3,224,224)
))