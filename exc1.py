from network.network import network
from layers.FClayer import FClayer
from layers.activationlayer import activationlayer
import numpy as np

def relu(z):
    if(z<0): return 0
    return z

def relu_prime(z):
    if(z<0): return 0
    return 1

def loss(y_true,y_hat):
    return 0.5*(y_true-y_hat)**2

def loss_prime(y_true,y_hat):
    return (y_true-y_hat)

x_train=np.array([[[0,0]],[[0,1]],[[1,0]],[[1,1]]])
y_train=np.array([[[0]],[[1]],[[1]],[[0]]])
net=network()
net.add(FClayer((1,2),(1,3)))
net.add(activationlayer((1,2),(1,3),relu,relu_prime))
net.add(FClayer((1,3),(1,1)))
net.add(activationlayer((1,3),(1,1),relu,relu_prime))

net.setuploss(loss,loss_prime)
net.fit(x_train,y_train,epochs=1000,learning_rate=0.01)
out= net.predict([[0,1]])
print(out)

