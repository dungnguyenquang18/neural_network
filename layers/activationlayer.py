from .layer import layer
import numpy as np

class activationlayer(layer):
    def _init_(self,input_shape,output_shape,activation,activation_prime):
        self.input_shape=input_shape
        self.output_shape=output_shape
        self.activaton=activation
        self.activaton_prime=activation_prime
        
    def forward_propagation(self,input):
        self.input=input
        self.output=self.activaton(input)               
        return self.output
    def backward_propagation(self,output_error,learning_rate):
        return self.activaton_prime(self.input)*output_error