import numpy as np

# TRAINING DATA ---------------------------------------------------------------------
data = np.array([[0,1],[1,0],[1,1],[2,0],[2,1],[1,5],[1,3],[5,2],[0,5],[3,2]]).T

# Classificaton (expected results)
t = np.array([1,1,1,2,2,5,3,5,5,3])

# PARAMETERS -----------------------------------------------------------------------

# Number of iterations
niter = 100000

# Learning rate
rate=1

# Number of classes
k = 6

# Size of each layers
layersizes = np.array([np.shape(data)[0],7,k])

#------------------------------------------------------------------------------------

nlayers = len(layersizes) - 1
ndata =  np.shape(data)[1]

# Weights and bias initialization
W = []
B = []
for i in range(nlayers):
     W.append(2*np.random.random((layersizes[i+1],layersizes[i])) - 1)
     B.append((2*np.random.random((layersizes[i+1],1))-1).dot(np.ones((1,ndata))))


# Learning iterations ---------------------------------------------------------------

tn = np.zeros((k,ndata))
for i in range(ndata):
    tn[t[i],i]=1


# X[l] is the output of layer l
X = [None]*(nlayers+1)
X[0] = data

# E[-(l+1)] is the part of the gradient that is recurent
E = [None]*(nlayers)

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
    
# Try the network:
print(X[-1])
for i in range(ndata):
    classe = np.argsort(X[-1][:,i])
    classe = classe[::-1]
    prob = X[-1][classe,i]*100
    print("class".rjust(5), "confidence".rjust(12), "expected:",t[i])
    for j in range(k):
        print('{0:5d} {1:12.3f}'.format(int(classe[j]),prob[j]))

print("Final mean square error", np.sum((tn - X[-1])**2)/ndata)





