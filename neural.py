import numpy as np
from PIL import Image as pil
import matplotlib.pyplot as plt
import glob
import pickle
import time

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

def read_data_emotions(folder, frac_test, frac_valid):
    data = []
    tdata = []
    vdata = []
    t = np.array([],np.int64)
    tt = np.array([],np.int64)
    vt = np.array([],np.int64)

    peoplelist = glob.glob(folder + '/*/')
    idx = np.random.choice(np.arange(0,len(peoplelist)), int(len(peoplelist)*(frac_test+frac_valid)), replace=False)
    n_test = int(frac_test*len(peoplelist))
    idxtest = idx[:n_test]
    idxvalid = idx[n_test:]
    
    for i in range(2):
        if i == 0:
            emo = 'happy'
        elif i == 1:
            emo = 'neutral'
        # elif i == 2:
        #     emo = 'sad'
        # elif i == 3:
        #     emo = 'angry'
        for j,person in enumerate(peoplelist):
            filelist = glob.glob(person + '/*'+emo+'_open_2.pgm')
            if len(filelist) > 0:
                alldata = read_pgm_p5_list(filelist)
                if any(idxtest == j):
                    tdata.append(alldata)
                    tt = np.append(tt,np.ones(len(filelist),np.int64)*i)
                elif any(idxvalid == j):
                    vdata.append(alldata)
                    vt = np.append(vt,np.ones(len(filelist),np.int64)*i)
                else:        
                    data.append(alldata)
                    t = np.append(t,np.ones(len(filelist),np.int64)*i)
    data = np.concatenate(data,axis=1)
    tdata = np.concatenate(tdata,axis=1)
    vdata = np.concatenate(vdata,axis=1)
    
    # # Shuffle data
    # shuffleidx = np.random.choice(np.arange(0,len(t)), len(t), replace=False)
    
    # data = data[:,shuffleidx]
    # t = t[shuffleidx]

    # n_test = int(len(t)*frac_test)
    
    # testing_data = data[:,:n_test]
    # testing_t = t[:n_test]
    
    # training_data = data[:,n_test:]
    # training_t = t[n_test:]

    pickle.dump((data, t), open("training.dat","wb"))
    pickle.dump((tdata, tt), open("testing.dat","wb"))
    pickle.dump((vdata, vt), open("validation.dat","wb"))

    return (data, t, tdata, tt, vdata, vt)

def read_data_people(folder, frac_test, frac_valid):
    data = []
    tdata = []
    vdata = []
    t = np.array([],np.int64)
    tt = np.array([],np.int64)
    vt = np.array([],np.int64)

    peoplelist = glob.glob(folder + '/*/')
        
    for j in range(len(peoplelist)):
        filelist = glob.glob(peoplelist[j] + '/*_2.pgm')
        alldata = read_pgm_p5_list(filelist)
        data.append(alldata)
        t = np.append(t,np.ones(len(filelist),np.int64)*j)

    data = np.concatenate(data,axis=1)
    
    # Shuffle data
    shuffleidx = np.random.choice(np.arange(0,len(t)), len(t), replace=False)
    
    data = data[:,shuffleidx]
    t = t[shuffleidx]

    n_test = int(len(t)*frac_test)
    n_valid = int(len(t)*frac_valid)
    
    tdata = data[:,:n_test]
    tt = t[:n_test]

    data = data[:,n_test:]
    t = t[n_test:]
    
    vdata = data[:,:n_valid]
    vt = t[:n_valid]
    
    data = data[:,n_valid:]
    t = t[n_valid:]

    pickle.dump((data, t), open("training.dat","wb"))
    pickle.dump((tdata, tt), open("testing.dat","wb"))
    pickle.dump((vdata, vt), open("validation.dat","wb"))

    return (data, t, tdata, tt, vdata, vt)


def example_data():
    # Example: Finding the max
    data = []
    t = []
    for i in range(3):
        for j in range(3):
            data.append([i,j])
            t.append(max(i,j))

    return (np.array(data).T, np.array(t))

def calc_performance(X,t):
    p=0
    for i in range(len(t)):
        if np.argmax(X[:,i]) == t[i]:
            p += 1
    return p/len(t) 

def example_data2():
    # Example: Finding the max
    data = np.random.random((3840,95))*155
    t = np.floor(np.random.random(95)*2).astype(np.int64)
    
    return data,t

def use_net(data,W,B):

    nlayers = len(W)
    ndata =  np.shape(data)[1]
    
    X = [None]*(nlayers+1)
    X[0] = data
    
    for j in range(nlayers):
        X[j+1] = 1/(1 + np.exp(-(W[j].dot(X[j])+B[j].dot(np.ones((1,ndata))))))

    return X[-1]/np.sum(X[-1],0)

def train_net(data,t,vdata,vt,tdata,tt,layersizes,rate,lam,niter):

    plt.ion()
    
    nlayers = len(layersizes) - 1
    nclass = max(t)+1
    
    layersizes = np.hstack([np.shape(data)[0],layersizes,nclass])
    ndata =  np.shape(data)[1]
    vndata = np.shape(vdata)[1]
    nlayers = len(layersizes) - 1

    # Setting the training data in the right format
    tn = np.zeros((nclass,ndata))
    for i in range(ndata):
        tn[t[i],i]=1

    vtn = np.zeros((nclass,vndata))
    for i in range(vndata):
        vtn[vt[i],i]=1

    # Weights and bias initialization
    W = []
    B = []
    
    for i in range(nlayers):
        W.append((2*np.random.random((layersizes[i+1],layersizes[i])) - 1))
        B.append((2*np.random.random((layersizes[i+1],1))-1)*0)
    
    
    # Learning iterations ---------------------------------------------------------------
    
    # X[l] is the output of layer l
    X = [None]*(nlayers+1)
    X[0] = data
    
    # E[-(l+1)] is the part of the gradient that is recurent
    E = [None]*(nlayers)
    
    grad = []
    perf_list = np.array([])
    vperf_list = np.array([])
    tperf_list = np.array([])
    mse_list = np.array([])
    mseprev = 2
    mse = 1
    for i in range(niter):

        it_time = time.time()
        
        # Propagation using f(u)=1/(1-exp(-u)) (sigmoid)
        for j in range(nlayers):
            X[j+1] = 1/(1 + np.exp(-(W[j].dot(X[j])+B[j].dot(np.ones((1,ndata))))))
    
        # Normalize output to get probability of each class
        #X[-1] = X[-1]/np.sum(X[-1],0)
    
        # Back propagation using f'(u) = f(u)**2 - f(u) (for the sigmoid)
        E[0] = ((tn-X[-1])*(X[-1]*(1-X[-1]))).T
    
        for j in range(nlayers-1):
            E[j+1] = E[j].dot(W[-(j+1)])*(X[-(j+2)]*(1-X[-(j+2)])).T
    
        for j in range(nlayers):
            W[-(j+1)] += rate*mse*(E[j].T.dot(X[-(j+2)].T)-lam*W[-(j+1)])
            B[-(j+1)] += rate*mse*E[j].T.dot(np.ones((ndata,1)))
            
        mse = np.sum((tn - X[-1])**2)/ndata
        
        if i%10==0:
            wsq = np.sum([np.sum(w**2) for w in W])
            err = mse + lam*wsq
            vX = use_net(vdata,W,B)
            tX = use_net(tdata,W,B)
            vperf = calc_performance(vX,vt)
            perf = calc_performance(X[-1]/np.sum(X[-1],0),t)
            tperf = calc_performance(tX,tt)
            perf_list = np.append(perf_list, perf)
            vperf_list = np.append(vperf_list, vperf)
            tperf_list = np.append(tperf_list, tperf)
            #mse_list = np.append(mse_list, mse)
            x = range(len(perf_list))
            it_time = time.time() - it_time
            print(i,err,mse,wsq,perf,vperf,tperf,it_time)
            plt.plot(x,perf_list,'r',x, vperf_list,'b',x, tperf_list,'g')
            plt.pause(0.05)

    mse = np.sum((tn - X[-1])**2)/ndata

    X[-1] = X[-1]/np.sum(X[-1],0)
    
    return W,B,X,mse

def print_performance(X,t):
    
    for i in range(np.shape(X)[1]):
        classe = np.argsort(X[:,i])
        classe = classe[::-1]
        prob = X[classe,i]*100
        print("class".rjust(5), "confidence".rjust(12), "expected:",t[i])
        for j in range(len(classe)):
            print('{0:5d} {1:12.3f}'.format(int(classe[j]),prob[j]))

def display_data(n,data,t,l,w):
    assert np.shape(data)[1] == len(t)
    for i in range(len(t)):
        if t[i] == n:
            plt.figure()
            plt.imshow(np.reshape(data[:,i],(l,w)))
            plt.title('t= %d, i = %d'%(t[i],i))
            

if __name__ == "__main__":

    # Number of iterations
    niter = 3000

    # Regularization
    lam = 0.005
    
    # Learning rate
    rate=0.02
    
    # Size of each hidden layer
    layersizes = np.array([150,75,25])
    
    try:
        (data,t) = pickle.load(open("training.dat","rb"))
        (tdata,tt) = pickle.load(open("testing.dat","rb"))
        (vdata,vt) = pickle.load(open("validation.dat","rb"))
    except:
        data,t,tdata,tt,vdata,vt = read_data_people('faces',0.2,0.2)

    # data,t = example_data2()
    # tdata,tt = example_data2()
    # vdata,vt = example_data2()

    
    W,B,Xi,mse = train_net(data,t,vdata,vt,tdata,tt,layersizes,rate,lam,niter)

    X = use_net(tdata,W,B)

    print_performance(X,tt)

    print("Overall performance of the net:", calc_performance(X,tt))
    print("final mse", mse)

