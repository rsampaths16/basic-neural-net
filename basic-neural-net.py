import numpy as np

def nonlin(x, deriv=False):
    if(deriv == True):
        return x*(1-x)

    return 1/(1+np.exp(-x))

#input data <- binary here
X = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
])

#output data <- xor here
y = np.array([
    [0],
    [1],
    [1],
    [0],
])

#deterministic random numbers
np.random.seed(1)

#synapses
syn0 = 2*np.random.random((3, 8)) - 1
syn1 = 2*np.random.random((8, 1)) - 1

#training step
epochs = 60000
alpha = 1 #learing rate
for j in range(epochs):
    #forward pass
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    
    # calculate error
    l2_error = y - l2
    if (j%10000) == 0:
        print "Error:", str(np.mean(np.abs(l2_error)))

    #backpropagation
    l2_delta = l2_error * nonlin(l2, deriv=True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1, deriv=True)

    #update weights
    syn1 += alpha * l1.T.dot(l2_delta)
    syn0 += alpha * l0.T.dot(l1_delta)

print "Output after training"
print l2
