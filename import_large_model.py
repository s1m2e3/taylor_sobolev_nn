import torchvision
import torch

#Assuming the device is a GPU named: 'cuda:0'

#Import resnet model defined by pytorch
big_model = torchvision.models.resnet18(num_classes=10)
#Load the pre-trained model from huggingface
# weights are found in: https://huggingface.co/bhumong/resnet18-cifar10/blob/main/README.md
big_model.load_state_dict(torch.load('./inputs/resnet18_cifar10.pth'))

