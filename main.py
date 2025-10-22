from import_large_model import import_large_model 
import torch
import tensorflow as tf

from training_loop import train
from testing_loop import test
from model import model
from torchvision import transforms


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Custom Dataset class to handle transformations


cifar = tf.keras.datasets.cifar10    
(x_train_full, y_train_full), (x_test, y_test) = cifar.load_data()
# Do not normalize here; ToTensor() will handle it.
x_valid, x_train = x_train_full[:5000], x_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2470, 0.2435, 0.2616)),
])

big_model = import_large_model()
big_model.to(device)
small_model = model(in_channels=3,pixel_size=32,num_classes=10).to(device)

train(big_model,small_model,x_train,y_train,x_valid,y_valid, preprocess=preprocess)
test(big_model,small_model,x_test,y_test)