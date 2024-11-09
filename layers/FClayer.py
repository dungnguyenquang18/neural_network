from .layer import layer
import numpy as np
class FClayer(layer):
    def _init_(self,input_shape,output_shape):
        self.input_shape=input_shape
        self.output_shape=output_shape
        self.weights=np.random.rand(input_shape[1],output_shape[1])-0.5
        self.bias=np.random.rand(1,output_shape)-0.5
    
    def forward_propagation(self,input):
        self.input=input
        self.output=np.dot(self.input,self.weights)+self.bias
        return self.output
    def backward_propagation(self,output_error,learning_rate):
        curr_ouput_error=np.dot(output_error,self.weights.T)
        dw=np.dot(self.input.T,self.output_error)
        self.weights-=dw*learning_rate
        self.bias-=learning_rate*output_error
        return curr_ouput_error
        