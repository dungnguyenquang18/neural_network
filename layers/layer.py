from abc import abstractmethod

class layer:
    def __init__(self) :
        self.input=None
        self.output=None
        self.output_shape=None
        self.input_shape=None
        raise NotImplementedError
    @abstractmethod
    def input(self):
         return self.input    
    @abstractmethod
    def output(self):
        return self.output   
    @abstractmethod
    def output_shape(self):
        return self.output_shape 
    @abstractmethod
    def input_shape(self):
        return self.input_shape  
    @abstractmethod
    def forward_propagation(self,input):
        raise NotImplementedError
    @abstractmethod 
    def backward_propagation(self,ouput_error,learning_rate):
        raise NotImplementedError