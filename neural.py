import numpy as np
from PIL import Image as pil
import matplotlib.pyplot as plt
import glob
import pickle

def read_pgm_p2(name):
    with open(name) as f:
        lines = f.readlines()

    # Ignores commented lines
    for l in list(lines):
        if l[0] == '#':
            lines.remove(l)
            
    # Makes sure it is ASCII format (P2)
    assert lines[0].strip() == 'P2' 

    # Converts data to a list of integers
    data = []
    for line in lines[1:]:
        data.extend([int(c) for c in line.split()])

    # Reshape data and normalizes it 
    return (np.array(data[3:]),(data[1],data[0]),data[2])

def read_pgm_p5_list(names):
    data = []
    for name in names:
        data.append(np.array(pil.open(name).getdata()))
    return np.array(data).T

def read_pgm_p2_list(names):
    data = []
    for name in names:
        data.append(read_pgm_p2(name)[0])
    return np.array(data).T

def read_data(folder, n_test):
    data = []
    t = np.array([])
    for i in range(4):
        if i == 0:
            emo = 'happy'
        elif i == 1:
            emo = 'sad'
        elif i == 2:
            emo = 'neutral'
        elif i == 3:
            emo = 'angry'
        filelist = glob.glob(folder + '/*/*'+emo+'_open.pgm')
        data.append(readdata_ascii(filelist))
        t = np.append(t,np.ones(len(filelist))*i)
        
    data = np.concatenate(data,axis=1)
    
    # Shuffle data
    shuffleidx = np.random.choice(np.arange(0,len(t)), len(t), replace=False)
    
    data = data[:,shuffleidx]
    t = t[shuffleidx]
    
    testing_data = data[:,:n_test]
    testing_t = data[:n_test]
    
    training_data = data[:,n_test:]
    training_t = t[n_test:]

    pickle.dump('training.dat',(training_data, training_t))
    pickle.dump('testing.dat',(testing_data, testing_t))

    return (training_data, training_t, testing_data, testing_t)

def example_data():
    # Example: Finding the max
    data = []
    t = []
    for i in range(3):
        for j in range(3):
            data.append([i,j])
            t.append(max(i,j))

    return (np.array(data).T, np.array(t))

def train_net(data,t,layersizes,rate,niter):

    nclass = max(t)+1
    layersizes = np.hstack([np.shape(data)[0],layersizes,nclass])
    ndata =  np.shape(data)[1]
    nlayers = len(layersizes) - 1
    
    # Weights and bias initialization
    W = []
    B = []
    for i in range(nlayers):
         W.append(2*np.random.random((layersizes[i+1],layersizes[i])) - 1)
         B.append((2*np.random.random((layersizes[i+1],1))-1).dot(np.ones((1,ndata))))
    
    
    # Learning iterations ---------------------------------------------------------------
    
    # Setting the training data in the right format
    tn = np.zeros((nclass,ndata))
    for i in range(ndata):
        tn[t[i],i]=1
    
    # X[l] is the output of layer l
    X = [None]*(nlayers+1)
    X[0] = data
    
    # E[-(l+1)] is the part of the gradient that is recurent
    E = [None]*(nlayers)
    
    grad = []
    
    for i in range(niter):
        
        # Propagation using f(u)=1/(1-exp(-u)) (sigmoid)
        for j in range(nlayers):
            X[j+1] = 1/(1 + np.exp(-(W[j].dot(X[j])+B[j])))
    
        # Normalize output to get probability of each class
        X[-1] = X[-1]/np.sum(X[-1],0)
    
        # Back propagation using f'(u) = f(u)**2 - f(u) (for the sigmoid)
        E[0] = ((tn-X[-1])*(X[-1]-X[-1]**2)).T
    
        for j in range(nlayers-1):
            E[j+1] = E[j].dot(W[-(j+1)])*(X[-(j+2)]-X[-(j+2)]**2).T
    
        for j in range(nlayers):
            W[-(j+1)] += rate*E[j].T.dot(X[-(j+2)].T)
            B[-(j+1)] += rate*E[j].T.dot(np.ones((ndata,ndata)))

    mse = np.sum((tn - X[-1])**2)/ndata
    
    return W,B,mse

def use_net(data,W,B):
    nlayers = len(W)
    
    X = [None]*(nlayers+1)
    X[0] = data
    
    for j in range(nlayers):
        X[j+1] = 1/(1 + np.exp(-(W[j].dot(X[j])+B[j])))

    return X[-1]/np.sum(X[-1],0)

def print_performance(X,t):
    
    for i in range(np.shape(X)[1]):
        classe = np.argsort(X[:,i])
        classe = classe[::-1]
        prob = X[classe,i]*100
        print("class".rjust(5), "confidence".rjust(12), "expected:",t[i])
        for j in range(len(classe)):
            print('{0:5d} {1:12.3f}'.format(int(classe[j]),prob[j]))
            

if __name__ == "__main__":

    # Number of iterations
    niter = 5000
    
    # Learning rate
    rate=1
    
    # Size of each hidden layer
    layersizes = np.array([11])

    data,t = example_data()

    W,B,mse = train_net(data,t,layersizes,rate,niter)

    X = use_net(data,W,B)

    print_performance(X,t)

