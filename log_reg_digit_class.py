#hw2_4.py was original filename


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
mnist_49_3000 = sio.loadmat('mnist_49_3000')
x = mnist_49_3000['x']
y = mnist_49_3000['y']
d,n= x.shape
#i = 0 #Index of the image to be visualized
#plt.imshow( np.reshape(x[:,i], (int(np.sqrt(d)),int(np.sqrt(d)))))
#plt.show()

sz = np.shape(x)
x = np.append(x,np.ones([1,3000]),0) # append a row of ones to the data matrix
y = np.where(y == -1,0,1) # find the indices of the labels where y = 1

#split the data into train and test sets
x_train = x[:785,:2000]
x_test = x[:785,2000:]

y_train = y[0,:2000]
y_test = y[0,2000:]

theta_old = np.zeros(785)
theta_new = np.ones(785) * 9999
obj_val = 0
gradient = np.ones(785)
hessian = np.zeros((785,785))
lmda = 10
flag1 = 0
threshold = 9999
stop_cond = 0

while np.linalg.norm(gradient) > 1e-8:
    if flag1 == 1:
        theta_min = theta_old
        theta_old = theta_new
        gradient = np.zeros(785)
        hessian = np.zeros((785,785))
    
    # calculate the gradient using the current value of theta
    for i in range(2000):
        x_i = x_train[:,i]
        t_dot_x = np.dot(x_i,-theta_old) # dot product of -theta and x_i
        z = 1 / (1+np.exp(t_dot_x)) # the value of eta at the using the result from line above
        gradient = x_i*(z - y[0,i]) + gradient
    
    gradient = gradient + theta_old* lmda * 2
    # calculate the hessian using the current value of theta
    for i in range(2000):
        x_i = x_train[:,i]
        t_dot_x = np.dot(x_i,-theta_old) # dot product of -theta and x_i
        z = np.exp(t_dot_x) / (1+np.exp(t_dot_x))**2 # the value of eta at the using the result from line above
        hessian = hessian + np.outer(x_i,x_i) * z

    hessian = hessian + 2*lmda*np.eye(785) # add the regularizing term
    inv_hessian = np.linalg.inv(hessian) # calc the inverse of the hessian
    theta_new = theta_old - np.dot(inv_hessian,gradient) # calc new theta 
    flag1 = 1

# calculating the objective function value at the minimizing theta
for i in range(2000):
    x_i = x_train[:,i]
    theta_min = -theta_new
    t_dot_x = np.dot(x_i,theta_min) # dot product of -theta and x_i
    z = 1 / (1+np.exp(t_dot_x)) # the value of eta at the using the result from line above
    obj_val = -( y[0,i]*np.log(z) + (1-y[0,i])*np.log(1-z) ) + obj_val
w = lmda*(np.linalg.norm(theta_new)**2) # regularizing term for objective function
obj_val = obj_val + lmda*(np.linalg.norm(theta_new)**2) #final objective function's value
print('objective value:')
print(obj_val)

predicted_labels = np.zeros(1000)
prob = np.zeros(50)
conf = np.zeros(50)
img_indx = np.zeros(50)
mislabeled_count = 0 # the number of images that are mislabeled

 # classify the test data using the value of theta found above
for i in range(1000):
    x_i = x_test[:,i]
    t_dot_x = np.dot(x_i,-theta_min) # dot product of -theta_min and x_i
    z = 1 / (1+np.exp(t_dot_x)) # the value of eta at the using the result from line above
    if z >= 0.5:
        predicted_labels[i] = 0 # record the value of the ith predicted label
        if predicted_labels[i] != y_test[i]: # compare predicted label with true label from test data
            mislabeled_count += 1
    else:
        predicted_labels[i] = 1 # record the value of the ith predicted label
        if predicted_labels[i] != y_test[i]: # compare predicted label with true label from test data
            mislabeled_count += 1
print('mislabeled count:')
print( mislabeled_count)
error = mislabeled_count/1000.0
print("error: ")
print(error)

# display 20 images with the highest prediction confidence in a 4x5 plot
'''
count = 0
conf = abs(prob - 0.5)/0.5
im2disp = np.zeros(20)
while count != 19:
    max_conf = max(conf)
    indx = np.where(conf == max_conf)
    print(indx)

    im2disp[count] = img_indx[indx]
    conf[indx] = 0
    count += 1
    
print(im2disp[19])
 '''
