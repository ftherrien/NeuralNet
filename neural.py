import numpy as np
from PIL import Image as pil
import matplotlib.pyplot as plt
import glob
import pickle
import time

def read_pgm_p2(name):
    """ Reads p2 type PGM files """
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
    """ Reads a list of p5 type PGM files """
    data = []
    for name in names:
        data.append(np.array(pil.open(name).getdata()))
    return np.array(data).T

def read_pgm_p2_list(names):
    """ Reads a list of p2 type PGM files"""
    data = []
    for name in names:
        data.append(read_pgm_p2(name)[0])
    return np.array(data).T

def read_data_emotions(folder, frac_test, frac_valid):
    """ Read the data and classifies it by emotion, selects random individual to be put in the validation and testing set"""
    # Initialization
    data = []
    tdata = []
    vdata = []
    t = np.array([],np.int64)
    tt = np.array([],np.int64)
    vt = np.array([],np.int64)

    # Finds the list of people
    peoplelist = glob.glob(folder + '/*/')
    idx = np.random.choice(np.arange(0,len(peoplelist)), int(len(peoplelist)*(frac_test+frac_valid)), replace=False)

    # Draws people from the list for the test set
    n_test = int(frac_test*len(peoplelist))
    idxtest = idx[:n_test]
    idxvalid = idx[n_test:]
    
    # Goes through all emotions
    for i in range(2):
        if i == 0:
            emo = 'happy'
        elif i == 1:
            emo = 'neutral'
        # elif i == 2:
        #     emo = 'sad'
        # elif i == 3:
        #     emo = 'angry'
        # Goes through the list of people
        for j,person in enumerate(peoplelist):
            filelist = glob.glob(person + '/*'+emo+'_open_2.pgm')
            if len(filelist) > 0:
                alldata = read_pgm_p5_list(filelist)
                # Add the right people to the right set
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

    # Saves the data
    pickle.dump((data, t), open("training.dat","wb"))
    pickle.dump((tdata, tt), open("testing.dat","wb"))
    pickle.dump((vdata, vt), open("validation.dat","wb"))

    return (data, t, tdata, tt, vdata, vt)

def read_data_people(folder, frac_test, frac_valid,face):
    """ Reads data and classifies it by person, selects random images and puts them in the training and testing sets"""
    data = []
    tdata = []
    vdata = []
    t = np.array([],np.int64)
    tt = np.array([],np.int64)
    vt = np.array([],np.int64)

    peoplelist = glob.glob(folder + '/*/')

    # Goes through all people
    for j in range(len(peoplelist)):
        filelist = glob.glob(peoplelist[j] + '/*' + face + '_2.pgm')
        alldata = read_pgm_p5_list(filelist)
        data.append(alldata) # Writes data
        t = np.append(t,np.ones(len(filelist),np.int64)*j) # Writes classes

    data = np.concatenate(data,axis=1)
    
    # Shuffle data
    shuffleidx = np.random.choice(np.arange(0,len(t)), len(t), replace=False)
    
    data = data[:,shuffleidx]
    t = t[shuffleidx]

    n_test = int(len(t)*frac_test)
    n_valid = int(len(t)*frac_valid)

    # Testing data
    tdata = data[:,:n_test]
    tt = t[:n_test]

    data = data[:,n_test:]
    t = t[n_test:]

    # Validation data
    vdata = data[:,:n_valid]
    vt = t[:n_valid]
    
    data = data[:,n_valid:]
    t = t[n_valid:]

    pickle.dump((data, t), open("training.dat","wb"))
    pickle.dump((tdata, tt), open("testing.dat","wb"))
    pickle.dump((vdata, vt), open("validation.dat","wb"))

    return (data, t, tdata, tt, vdata, vt)


def example_data():
    """ Example of a small set of data, for troubleshooting"""
    data = []
    t = []
    for i in range(3):
        for j in range(3):
            data.append([i,j])
            t.append(max(i,j))

    return (np.array(data).T, np.array(t))

def calc_performance(X,t):
    """ Calculates perforamnce of the network by comparing the class with the highest confidence level to the expected calss"""
    p=0
    for i in range(len(t)):
        if np.argmax(X[:,i]) == t[i]:
            p += 1
    return float(p)/len(t) 

def example_data2():
    """ Generates radom data of the right format, for troubleshooting """
    # Example: Finding the max
    data = np.random.random((3840,95))*155
    t = np.floor(np.random.random(95)*2).astype(np.int64)
    
    return data,t

def use_net(data,W,B):
    """ Uses the guven weights and bias to classify the data in "data", it outputs the normalized last layer"""

    nlayers = len(W)
    ndata =  np.shape(data)[1]
    
    X = [None]*(nlayers+1)
    X[0] = data

    # Propagation
    for j in range(nlayers):
        X[j+1] = 1/(1 + np.exp(-(W[j].dot(X[j])+B[j].dot(np.ones((1,ndata))))))

    return X[-1]/np.sum(X[-1],0)

def train_net(data,t,vdata,vt,layersizes,rate,lam,niter):
    """Trains the network on "data" comparaing it to "t", it returns the trained weights and bias and the values in all layers. The validation set "vdata" is only used for visual comparison"""

    # Display initialization
    plt.ion()
    plt.figure()
    plt.plot([],'r',label = 'Training data')
    plt.plot([],'b',label = 'Testing data')
    plt.xlabel('Epoch')
    plt.ylabel('Network Performance')
    plt.legend(loc=4)
    
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

    # Used for display
    perf_list = np.array([])
    vperf_list = np.array([])

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

        # Display updated every 10 iterations
        if i%10==0:
            wsq = np.sum([np.sum(w**2) for w in W])
            err = mse + lam*wsq

            # Assesssing the performance of the net on the validation data
            vX = use_net(vdata,W,B)
            vperf = calc_performance(vX,vt)
            perf = calc_performance(X[-1]/np.sum(X[-1],0),t)
            perf_list = np.append(perf_list, perf)
            vperf_list = np.append(vperf_list, vperf)

            # Printing and plotting the current state of the network
            x = np.arange(len(perf_list))*10
            it_time = time.time() - it_time
            print(i,err,mse,wsq,perf,vperf,it_time)
            plt.plot(x,perf_list,'r',label = 'Training data')
            plt.plot(x, vperf_list,'b',label = 'Testing data')
            plt.pause(0.05)

    mse = np.sum((tn - X[-1])**2)/ndata

    # Normalizes the last layer
    X[-1] = X[-1]/np.sum(X[-1],0)
    
    return W,B,X,mse

def print_performance(X,t):
    """ For each image in the testing set, prints the expected class end the level of confidence for each class """
    
    for i in range(np.shape(X)[1]):
        classe = np.argsort(X[:,i])
        classe = classe[::-1]
        prob = X[classe,i]*100
        print("class".rjust(5), "confidence".rjust(12), "expected:",t[i])
        for j in range(len(classe)):
            print('{0:5d} {1:12.3f}'.format(int(classe[j]),prob[j]))

def display_data(n,data,t,l,w):
    """ Reads a vector and displays it as an image, it will display all the image of class n in set "data" with target vector "t" """
    assert np.shape(data)[1] == len(t)
    for i in range(len(t)):
        if t[i] == n:
            plt.figure()
            plt.imshow(np.reshape(data[:,i],(l,w)))
            plt.title('t= %d, i = %d'%(t[i],i))
            

if __name__ == "__main__":

    # Number of iterations
    niter = 4000

    # Regularization
    lam = 0.005
    
    # Learning rate
    rate=0.008
    
    # Size of each hidden layer
    # Good performance at 150,75,25
    layersizes = np.array([150,75,25])

    # Re-use same data?
    use_old = False

    # Sunglasses in training data?
    sg_in_tr = False

    # Sunglasses in testing data?
    sg_in_t = False

    # Number of repetition to assess average performance
    n_stat = 1
    
    avg_train = []
    avg_test = []
    for i in range(n_stat):
            
        # Uses preloaded data in the current directory 
        if use_old:
            (data,t) = pickle.load(open("training.dat","rb"))
            (tdata,tt) = pickle.load(open("testing.dat","rb"))
            (vdata,vt) = pickle.load(open("validation.dat","rb"))
        else:
            # Read open and sunglasses data seperately
            data_sg,t_sg,tdata_sg,tt_sg,vdata_sg,vt_sg = read_data_people('faces',0.1,0.1, 'sunglasses')
            data,t,tdata,tt,vdata,vt = read_data_people('faces',0.1,0.1,'open')

            # Add sunglasses data if true
            if sg_in_tr:
                data = np.concatenate([data,data_sg],axis=1)
                t = np.concatenate([t, t_sg])
            if sg_in_t:
                tdata = np.concatenate([tdata,tdata_sg],axis=1)
                tt = np.concatenate([tt, tt_sg])
                vdata = np.concatenate([vdata,vdata_sg],axis=1)
                vt = np.concatenate([vt, vt_sg])
       
            pickle.dump((data, t), open("training.dat","wb"))
            pickle.dump((tdata, tt), open("testing.dat","wb"))
            pickle.dump((vdata, vt), open("validation.dat","wb"))
       
       
        W,B,Xi,mse = train_net(data,t,vdata,vt,layersizes,rate,lam,niter)
       
        X = use_net(tdata,W,B)
       
        print_performance(X,tt)

        avg_train.append(calc_performance(Xi[-1],t))
        avg_test.append(calc_performance(X,tt))
       
        print("Performance on the training set:", calc_performance(Xi[-1],t))
        print("Performance on the tresting set:", calc_performance(X,tt))
        print("final mse", mse)
    print('==============================')
    print('Average perfomance on the training set:', sum(avg_train)/n_stat)
    print('Average perfomance on the testing set:', sum(avg_test)/n_stat)
    print('==============================')
    print('Min and Max perfomance on the training set:',min(avg_train), max(avg_train))
    print('Min and Max perfomance on the testing set:',min(avg_test), max(avg_test))

    
       

