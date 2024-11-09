

class network:
    def _init_(self):
        self.layers=[]
        self.loss=None
        # self.loss=None
        self.loss_prime=None
        
    def add (self,layer):
        self.layers.append(layer)
        
    def setuploss(self,loss,loss_prime):
        self.loss=loss
        self.loss_prime=loss_prime
        
    def predict(self,input):
        res=[]
        n=len(input)
        for i in range(n):
            output=input[i]
            
            for layer in self.layers:
                output=layer.forward_propagation(output)
            res.append(output)
            
            
            
    def fit(self,x_train,y_train,learning_rate,epochs):
        n=len(x_train)
        for i in range (epochs):
            err=0
            for j in range(n):
                output=x_train[j]
                # forward
                for layer in self.layers:
                    output=layer.forward_propagation(output)
                err+=self.loss(y_train[j],output)
                # backward
                error=self.loss_prime(y_train[j],output)
                for layer in reversed(self.layers):
                    error=layer.backward_propagation(error,learning_rate)
            err/=n        
            print('epoch: %d/%d err %f'%(i,epochs,err))    