import numpy as np
import math
class LogisticRegression:


    def __init__(self, learning_rate, epoch, batch_size):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size


    def fit(self,X,y):
        
        i=0
        nsamples, nx, ny = X.shape
        train_reshaped = X.reshape((nsamples,nx*ny))
        nsamples, col = train_reshaped.shape
        

        tutucu = np.zeros((nsamples, 10))
        tutucu[np.arange(nsamples), y] = 1
        one_hot_label = tutucu


        param = self.initilize_weight_and_bias(train_reshaped)
        weight = param['w']
        bias = param['b']
 
        batch = 0
        j=0

        while i < self.epoch:
            while j < nsamples:

                section_train = train_reshaped[j:j+self.batch_size,:]
                section_label = one_hot_label[j:j+self.batch_size,:]
                logit = np.dot(section_train,weight) + bias
                softmax = self.softmax(logit)
                weight =  weight - self.learning_rate * np.dot(section_train.T, (softmax - section_label))
                bias = bias - self.learning_rate * np.sum(softmax - section_label)
                
                j=j+self.batch_size
            i=i+1
        
        return weight,bias

    def predict(self,X,label,weight,bias):

        nsamples, nx, ny = X.shape
        test_reshaped = X.reshape((nsamples,nx*ny))

        logit = np.dot(test_reshaped,weight) + bias
        softmax = self.softmax(logit)

        tutucu = softmax.max(axis=1).reshape(-1, 1)
        softmax[:] = np.where(softmax == tutucu, 1, 0)

        k = np.zeros((nsamples, 10))
        k[np.arange(nsamples), label] = 1
        one_hot_label = k

        return softmax,one_hot_label

                
    def initilize_weight_and_bias(self,train):
        nsamples, imsize = train.shape
        #weight = [[0] * col for i in range(10)]
        weight = np.random.rand(imsize, 10) #(784*10)
        shape = (1,10)
        bias = np.zeros(shape) # (10*1) 
        param = {
            'w' : weight, # (10*784)
            'b' : bias  # (10*1)
        }
        return param


    def softmax(self,logits):
        sum = 0
        row, col = logits.shape
        shape = (row,col)
        S = np.zeros(shape) 

        for i in range(row):
            sum = np.sum(math.e ** logits[i,:])
            for j in range(col):
                S[i,j] = (math.e ** logits[i,j]) / sum
        

        return S

    
        