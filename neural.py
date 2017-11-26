import numpy as np

data = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ]).T

# Classificaton
t = np.array([[0,1,1,0]])

# Sieze of each layers, layersizes[-1] = 1
layersizes = np.array([np.shape(data)[0],4,1])
nlayers = len(layersizes) - 1

# Data
X = [None]*(nlayers+1)
X[0] = data

# Weights initialization
W = []
E = [None]*(nlayers)
# for i in range(nlayers):
#     W.append(2*np.random.random((layersizes[i+1],layersizes[i])) - 1)

W = [np.array([[-0.91284062, -0.67858789, -0.21414769],
               [-0.64454472,  0.07708981, -0.87895268],
               [-0.66035884,  0.78464445, -0.01624236],
               [-0.14250009,  0.78994748, -0.3534058 ]]),
     np.array([[-0.26415044,  0.3119735 , -0.47330244,  0.85508137]])]

print(W)

# Learning iterations
rate=1
for i in range(2):
    # Propagation using f(u)=1/(1-exp(-u))
    for j in range(nlayers):
        X[j+1] = 1/(1 + np.exp(-W[j].dot(X[j])))

    print('X ==============================================')
    print(X[0].T)
    print(X[1].T)
    print(X[2].T)
    # Back propagation using f'(u) = f(u)**2 - f(u) (in this case)
    E[0] = ((t-X[-1])*(X[-1]-X[-1]**2)).T
    # W[-1] += rate*E[0].T.dot(X[-2].T) # New method?
    for j in range(nlayers-1):
        E[j+1] = E[j].dot(W[-(j+1)])*(X[-(j+2)]-X[-(j+2)]**2).T
        # W[-(j+2)] += E[j+1].T.dot(X[-(j+3)].T) # New method?

    for j in range(nlayers):
        W[-(j+1)] += rate*E[j].T.dot(X[-(j+2)].T)

    print('E ++++++++++++++++++++++++++++++++++++++++++++++')
    print(E[0])
    print(E[1])
        
    print('WEIGHTS -----------------------------------------')
    print(W[0].T)
    print(W[1].T)
    


print(W)


