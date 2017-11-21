 # -*- coding: utf-8 -*-
"""
A DIY Neural Net code for  basic concepts

@author: whatdhack

Download MNIST csv data from https://pjreddie.com/projects/mnist-in-csv/

Ref:  
    1. The Deep Learning Book by Ian Goodfellow, Yoshua Bengio and Aaron Courville
         (http://www.deeplearningbook.org/)
    2. Make Your Own Neural Networks by Tariq Rashid 
         (https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork )
    3. Machine Learning by Andrew Ng on Coursera 
         (https://www.coursera.org/learn/machine-learning)

"""


import numpy as np
# scipy.special for the sigmoid function expit()
import scipy.special

from random import shuffle
import time


# neural network class definition
class NeuralNetwork:
    
    
    # initialise the neural network
    def __init__(self, layernodes, learningrate, mbsize=3, training_data_file_name="data/mnist_train.csv", test_data_file_name="data/mnist_test.csv"):
        # set number of layers , nodss per layer
        self.layernodes = layernodes
        self.numlayers = len(layernodes)
        # link weight matrices, w
        self.w = []
        for i in  range (len(layernodes) -1) :
            self.w.append (np.random.normal(0.0, pow(self.layernodes[i], -0.5), (self.layernodes[i+1], self.layernodes[i])))
            print ('w[%d].shape'% i, self.w[i].shape)

        # learning rate
        self.lr = learningrate
        
        # mini-batch size 
        self.mbsize = mbsize 
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        
        # load the  training data CSV file into a list
        training_data_file = open(training_data_file_name, 'r')
        self.training_data = training_data_file.readlines()
        training_data_file.close()
        
        # shuffle the trainign data 
        shuffle(self.training_data)
        
        # load the mnist test data CSV file into a list
        test_data_file = open(test_data_file_name, 'r')
        self.test_data = test_data_file.readlines()
        test_data_file.close()
        
        self.train = self.trainMBSGD
        
        return

    
    # forward pass through  the neural network
    #    Ni  Wi Nj  Wj  Nk
    #  Ii  Oi Ij  Oj  Ik  Ok
    def forward (self, inputs_list):
        # convert inputs list to 2d array
        # calculate signals into a  layer
        # calculate the signals emerging from a layer

        inputsl  = [np.array(inputs_list, ndmin=2).T]
        outputsl = [np.array(inputs_list, ndmin=2).T]
        for l  in range(1,self.numlayers):
            inputsl.append(np.dot(self.w[l-1], outputsl[l-1]))
            outputsl.append(self.activation_function(inputsl[l]))
            
        return outputsl
    
    
    # brack prop errors and weight adjustment the neural network
    # Ni Wi Nj Wj Nk
    # Ei    Ej    Ek
    def backpropSGD(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        targets = np.array(targets_list, ndmin=2).T
        outputsl = self.forward(inputs_list)
        errorsl = [None]*(self.numlayers)
        errorsl[-1] = targets - outputsl[-1]

        #print ('range(self.numlayers-2,0,-1)', range(self.numlayers-2,0,-1))
        for l in range(self.numlayers-2,-1,-1):# W1,W0
            errorsl[l] = np.dot(self.w[l].T,  errorsl[l+1]) # error bp is proportinal to weight 
            
            errorp = errorsl[l+1] * outputsl[l+1] * (1.0 - outputsl[l+1]) # dE/dw = E*do/dw for sigmoid
            opprev = np.transpose(outputsl[l])
            dw = self.lr * np.dot( errorp,  opprev)
            #print ('w[l].shape', self.w[l].shape, 'dw.shape', dw.shape)
            self.w[l] += dw
            #print ('w[%d]'%l, self.w[l])   
            
    # brack prop errors and weight adjustment the neural network
    # Ni Wi Nj Wj Nk
    # Ei    Ej    Ek
    def backprop1a(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        targets = np.array(targets_list, ndmin=2).T
        outputsl = self.forward(inputs_list)
        errorsl = [None]*(self.numlayers)
        errorsl[-1] = targets - outputsl[-1]

        #print ('range(self.numlayers-2,0,-1)', range(self.numlayers-2,0,-1))
        dwl = []
        for l in range(self.numlayers-2,-1,-1):# W1,W0
            errorsl[l] = np.dot(self.w[l].T,  errorsl[l+1])
            
            errorp = errorsl[l+1] * outputsl[l+1] * (1.0 - outputsl[l+1])
            opprev = np.transpose(outputsl[l])
            dw = self.lr * np.dot( errorp,  opprev)
            #print ('w[l].shape', self.w[l].shape, 'dw.shape', dw.shape)
            #self.w[l] += dw
            dwl.append(dw)
            
        return dwl
    
    # brack prop errors and weight adjustment the neural network
    # Ni Wi Nj Wj Nk
    # Ei    Ej    Ek
    def backpropMBSGD(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        targets = np.array(targets_list, ndmin=2).T
        outputsl = self.forward(inputs_list)
        errorsl = [None]*(self.numlayers)
        errorsl[-1] = targets - outputsl[-1]

        errl = []
        opprevl = []
        for l in range(self.numlayers-2,-1,-1):# W1,W0
            errorsl[l] = np.dot(self.w[l].T,  errorsl[l+1])
            
            errorp = errorsl[l+1] * outputsl[l+1] * (1.0 - outputsl[l+1])
            opprev = outputsl[l]
            #print ('errorp.shape', errorp.shape, 'opprev.shape', opprev.shape)
            errl.append(errorp)
            opprevl.append(opprev)
            
        return errl, opprevl

    
    
    # train the neural network with SGD and minibatch size 1 
    def trainSGD(self, numepochs):
        # epochs is the number of times the training data set is used for training
        
        
        for e in range(numepochs):
            # go through all records in the training data set
            
            starttime = time.time()
            for record in self.training_data:
                # split the record by the ',' commas
                all_values = record.split(',')
                # scale and shift the inputs
                inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                # create the target output values (all 0.01, except the desired label which is 0.99)
                targets = np.zeros(self.layernodes[-1]) + 0.01
                # all_values[0] is the target label for this record
                targets[int(all_values[0])] = 0.99
                self.backpropSGD(inputs, targets)
                
            print ('epoch ', e, 'completed in %d s'%(time.time()- starttime))
        

    # train the neural network with SGD and minibatch not 1 
    def trainMBSGD(self, numepochs):
        # epochs is the number of times the training data set is used for training
        
        minibatch_size  = self.mbsize
        for e in range(numepochs):
            # go through all records in the training data set
            
            num_minibatch = int (len(self.training_data) / minibatch_size)
            if num_minibatch > 100000:
                num_minibatch = 100000
            starttime = time.time()
            for i in range(0, num_minibatch*minibatch_size, minibatch_size):
            
                errlmb = []
                opprevlmb=[]
                for record in self.training_data[i:i+minibatch_size]:
                    # split the record by the ',' commas
                    all_values = record.split(',')
                    # scale and shift the inputs
                    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                    # create the target output values (all 0.01, except the desired label which is 0.99)
                    targets = np.zeros(self.layernodes[-1]) + 0.01
                    # all_values[0] is the target label for this record
                    targets[int(all_values[0])] = 0.99
                    errl, opprevl = self.backpropMBSGD(inputs, targets)
                    errlmb.append(errl)
                    opprevlmb.append(opprevl)
                
                errnd = np.array(errlmb)
                errf = np.mean(errnd, axis=0)
                
                opprevnd = np.array(opprevlmb)
                opprevf = np.mean(opprevnd, axis=0)
    
                
                for l in range ( len(self.w) ):
                    l1 = len(self.w)-1-l
                    dw = self.lr * np.dot( errf[l1],    np.transpose(opprevf[l1]))
                    self.w[l] += dw
                    # self.w[l] += dwf[len(self.w)-1-l]
                                   
            print ('epoch ', e, 'completed in %d s'%(time.time()- starttime))
            

    # run test 
    def test(self):
        # test the neural network
        
        # score for how well the network performs, initially empty
        score = []
        
        # go through all the records in the test data set
        for record in self.test_data:
            # split the record by the ',' commas
            all_values = record.split(',')
            # correct answer is first value
            correct_label = int(all_values[0])
            # scale and shift the inputs
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # forward  pass through the  network
            outputsl = self.forward (inputs)
            # the index of the highest value corresponds to the label
            label = np.argmax(outputsl[-1])
            # append correct or incorrect to list
            if (label == correct_label):
                # network's answer matches correct answer, add 1 to score
                score.append(1)
            else:
                # network's answer doesn't match correct answer, add 0 to score
                score.append(0)
                pass
            
            pass
        
        # calculate the performance score, the fraction of correct answers
        print ("performance = ", np.asarray(score).mean())
        
        
        

def main():
    # number of layers and nodes in each layer
    # input, hidden, output
    layernodes = [784,200,10]
    
    # learning rate
    learning_rate = 0.1 
    
    #mini-batch size
    mbsize = 3
    
    # download MNIST csv data from https://pjreddie.com/projects/mnist-in-csv/
    # create the  neural network
    n = NeuralNetwork(layernodes, learning_rate, mbsize, "data/mnist_train.csv", "data/mnist_test.csv")
    
    # train the neural network
    n.train(5)
    
    # test the neural network
    n.test()

    
if __name__ == '__main__':
    start = time.time()
    main()
    print ( 'compeleted in %d s'%(time.time()- start))