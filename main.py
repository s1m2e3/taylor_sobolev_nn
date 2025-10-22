from import_large_model import import_large_model 
import torch
import tensorflow as tf
from training_loop import train
from testing_loop import test
from model import model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

cifar = tf.keras.datasets.cifar10    
(x_train_full, y_train_full), (x_test, y_test) = cifar.load_data()
x_train_full = x_train_full / 255.0
x_test = x_test / 255.0
x_valid, x_train = x_train_full[:5000], x_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
x_valid_tensor = torch.tensor(x_valid, dtype=torch.float32).to(device)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.long).to(device)
big_model = import_large_model()
big_model.to(device)
small_model = model(in_channels=x_train_tensor.shape[3],pixel_size=32,num_classes=10).to(device)

train(big_model,small_model,x_train_tensor,y_train_tensor,x_valid_tensor,y_valid_tensor)
test(big_model,small_model,x_test_tensor,y_test)