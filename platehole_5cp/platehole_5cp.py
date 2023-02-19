#import packages
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import sciann as sn
import pandas as pd
from numpy.random import seed
import random
import scipy
import os, sys, time
import shutil #to copy source file into result folder
import tensorflow as tf

#create output folder if it does not exist
if not os.path.isdir('./output'):
    os.mkdir('./output')

#delete content of output folder if it exists
for f in os.listdir('./output'):
    os.remove(os.path.join('./output', f))
    
wd = os.getcwd()    #get current working directory
shutil.copy(os.path.realpath(__file__), wd+'/output')    #copy source to output folder

#elasticity parameters
E = 210.
nu = 0.3

#import shape parameters
P = []
with open('./C1/CPs.txt', 'r') as arch:
    for line in arch:
        P.append(eval(line.rstrip()))
        
nr = np.shape(P)[0] #memorize number of rows in R_len
P_row = np.reshape(P,(8*nr),order='F') #write R on a single row

P1 = P_row[0:nr]         #vector containing values of first CP
P2 = P_row[nr:2*nr]      #vector containing values of second CP
P3 = P_row[2*nr:3*nr]      #vector containing values of third CP
P4 = P_row[3*nr:4*nr]      #vector containing values of fourth CP
P5 = P_row[4*nr:5*nr]      #vector containing values of fifth CP
P6 = P_row[5*nr:6*nr]      #vector containing values of sixth CP
P7 = P_row[6*nr:7*nr]      #vector containing values of seventh CP
P8 = P_row[7*nr:8*nr]      #vector containing values of eigth CP

P1 = (P1-np.min(P1))/(np.max(P1)-np.min(P1)) #normalize input P1
P2 = (P2-np.min(P2))/(np.max(P2)-np.min(P2)) #normalize input P2
P3 = (P3-np.min(P3))/(np.max(P3)-np.min(P3)) #normalize input P3
P4 = (P4-np.min(P4))/(np.max(P4)-np.min(P4)) #normalize input P4
P5 = (P5-np.min(P5))/(np.max(P5)-np.min(P5)) #normalize input P5
P6 = (P6-np.min(P6))/(np.max(P6)-np.min(P6)) #normalize input P6
P7 = (P7-np.min(P7))/(np.max(P7)-np.min(P7)) #normalize input P7
P8 = (P8-np.min(P8))/(np.max(P8)-np.min(P8)) #normalize input P8

#choose how many and which geometries use for training.
nds = 100 #number of training geometries
train_indx = np.arange(0,nds) #indices of training geometries. Cardinality must be nds

#build training dataset
num = str(train_indx[0]+1) #+1 because naming of datasets starts at 1, not 0
dataset1 = np.array(pd.read_csv('./C1/dataOmega'+num+'.csv')).astype(np.float32)
x_r = dataset1[:,0] #vector coordinates single geometry, used for dimensioning
N1 = np.size(x_r) #number of rows of a single geometry

dataset = dataset1 #initialize dataset
for i in range(1,nds):
  num = str(train_indx[i]+1)
  dataset=np.append(dataset,np.array(pd.read_csv('./C1/dataOmega'+num+'.csv')).astype(np.float32))
dataset = np.reshape(dataset,(N1*nds,17))

#coordinates in parametric domain, do not change with physical shape
x_data = dataset[:,0]
y_data = dataset[:,1]
N = np.size(x_data)

#data on displacements and stresses
ux_data = dataset[:,4]
uy_data = dataset[:,5]
sigmaxx_data = dataset[:,9]
sigmayy_data = dataset[:,10]
sigmaxy_data = dataset[:,11]

#define body forces (if present)
fx_data = 0.
fy_data = 0.

#import data of inverse Jacobian
xix_data = dataset[:,12]
xiy_data = dataset[:,13]
etax_data = dataset[:,14]
etay_data = dataset[:,15]

#define shape data, i.e. training data of the control points of training geometries
A1_data = P1[train_indx[0]]*np.ones_like(x_r)
for i in range(1,nds):
  A1_data = np.append(A1_data,P1[train_indx[i]]*np.ones_like(x_r))
  
A2_data = P2[train_indx[0]]*np.ones_like(x_r)
for i in range(1,nds):
  A2_data = np.append(A2_data,P2[train_indx[i]]*np.ones_like(x_r))
  
A3_data = P3[train_indx[0]]*np.ones_like(x_r)
for i in range(1,nds):
  A3_data = np.append(A3_data,P3[train_indx[i]]*np.ones_like(x_r))
  
A4_data = P4[train_indx[0]]*np.ones_like(x_r)
for i in range(1,nds):
  A4_data = np.append(A4_data,P4[train_indx[i]]*np.ones_like(x_r))
  
A5_data = P5[train_indx[0]]*np.ones_like(x_r)
for i in range(1,nds):
  A5_data = np.append(A5_data,P5[train_indx[i]]*np.ones_like(x_r))
  
A6_data = P6[train_indx[0]]*np.ones_like(x_r)
for i in range(1,nds):
  A6_data = np.append(A6_data,P6[train_indx[i]]*np.ones_like(x_r))
  
A7_data = P7[train_indx[0]]*np.ones_like(x_r)
for i in range(1,nds):
  A7_data = np.append(A7_data,P7[train_indx[i]]*np.ones_like(x_r))
  
A8_data = P8[train_indx[0]]*np.ones_like(x_r)
for i in range(1,nds):
  A8_data = np.append(A8_data,P8[train_indx[i]]*np.ones_like(x_r))

#######################
# Neural Network Setup.
#######################
dtype='float32'

#define the variables of the neural network
x = sn.Variable("x", dtype=dtype)
y = sn.Variable("y", dtype=dtype)
A1 = sn.Variable("A1", dtype=dtype)
A2 = sn.Variable("A2", dtype=dtype)
A3 = sn.Variable("A3", dtype=dtype)
A4 = sn.Variable("A4", dtype=dtype)
A5 = sn.Variable("A5", dtype=dtype)
A6 = sn.Variable("A6", dtype=dtype)
A7 = sn.Variable("A7", dtype=dtype)
A8 = sn.Variable("A8", dtype=dtype)
xix = sn.Variable("xix", dtype=dtype)
xiy = sn.Variable("xiy", dtype=dtype)
etax = sn.Variable("etax", dtype=dtype)
etay = sn.Variable("etay", dtype=dtype)

#define the functionals of the neural network
Uxy = sn.Functional("Uxy", [x, y, A1, A2, A3, A4, A5, A6, A7, A8], 5*[20], 'tanh')
Vxy = sn.Functional("Vxy", [x, y, A1, A2, A3, A4, A5, A6, A7, A8], 5*[20], 'tanh')
Sxx = sn.Functional("Sxx", [x, y, A1, A2, A3, A4, A5, A6, A7, A8], 5*[20], 'tanh')
Syy = sn.Functional("Syy", [x, y, A1, A2, A3, A4, A5, A6, A7, A8], 5*[20], 'tanh')
Sxy = sn.Functional("Sxy", [x, y, A1, A2, A3, A4, A5, A6, A7, A8], 5*[20], 'tanh')

#compute elasticity parameters
C11 = E/(1-nu**2)
C12 = nu*E/(1-nu**2)
C33 = E*(1-nu)/(2*(1-nu**2))

#compute deformations (using chain rule) for residual computations
Exx = sn.diff(Uxy, x)*xix + sn.diff(Uxy, y)*etax
Eyy = sn.diff(Vxy, x)*xiy + sn.diff(Vxy, y)*etay
Gxy = sn.diff(Uxy, x)*xiy + sn.diff(Uxy, y)*etay + sn.diff(Vxy, x)*xix + sn.diff(Vxy, y)*etax

#compute residuals
c1 = sn.Tie(Sxx, Exx*C11 + Eyy*C12)
c2 = sn.Tie(Syy, Eyy*C11 + Exx*C12)
c3 = sn.Tie(Sxy, Gxy*C33)

#assign data
Du = sn.Data(Uxy)
Dv = sn.Data(Vxy)
Ds1 = sn.Data(Sxx)
Ds2 = sn.Data(Sxy)
Ds3 = sn.Data(Syy)

targets = [Du, Dv, Ds1, Ds2, Ds3, c1, c2, c3]

#define scimodel
m = sn.SciModel([x, y, A1, A2, A3, A4, A5, A6, A7, A8, xix, xiy, etax, etay], targets)

#define residual data (zero for zero residual)
c1_data = np.zeros_like(y_data)
c2_data = np.zeros_like(y_data)
c3_data = np.zeros_like(y_data)

#train model
h = m.train([x_data, y_data, A1_data, A2_data, A3_data, A4_data, A5_data, A6_data, A7_data, A8_data, xix_data, xiy_data, etax_data, etay_data], [ux_data, uy_data, sigmaxx_data, sigmaxy_data, sigmayy_data, c1_data, c2_data, c3_data], batch_size=100, epochs=1200,  learning_rate = ([0, 300, 700, 1100], [0.001, 0.0005, 0.0002, 0.0001]), log_parameters={}, adaptive_weights={'method':'NTK', 'freq': 100}, verbose = 2)


#########################################################################
##########                 END OF PINN MODEL                  ###########
##########     GENEREATION AND PRINTING OF RESULTS BEGINS     ###########
#########################################################################

#save the weights of the trained PINN
m.save_weights('./output/weights_PINN.ckpt')
#copy output (if it was created) to output folder
#shutil.copy('./terminal_output.txt', './output/terminal_output.txt')

#save the plot of the total loss function
fig = plt.figure()
new_plot = fig.add_subplot(111)
new_plot.plot(h.history['loss'], label='loss')
plt.yscale('log')
plt.legend()
fig.savefig('./output/loss.png')

#save the plots of the partial loss functions for displacements and stresses
fig = plt.figure()
new_plot = fig.add_subplot(111)
new_plot.plot(h.history['Uxy_loss'], label='ux loss')
plt.yscale('log')
plt.legend()
fig.savefig('./output/ux_loss.png')

fig = plt.figure()
new_plot = fig.add_subplot(111)
new_plot.plot(h.history['Vxy_loss'], label='uy loss')
plt.yscale('log')
plt.legend()
fig.savefig('./output/uy_loss.png')

fig = plt.figure()
new_plot = fig.add_subplot(111)
new_plot.plot(h.history['Sxx_loss'], label='Sxx loss')
plt.yscale('log')
plt.legend()
fig.savefig('./output/Sxx_loss.png')

fig = plt.figure()
new_plot = fig.add_subplot(111)
new_plot.plot(h.history['Sxy_loss'], label='Sxy loss')
plt.yscale('log')
plt.legend()
fig.savefig('./output/Sxy_loss.png')

fig = plt.figure()
new_plot = fig.add_subplot(111)
new_plot.plot(h.history['Syy_loss'], label='Syy loss')
plt.yscale('log')
plt.legend()
fig.savefig('./output/Syy_loss.png')

#save the plot of the normalized total loss function
loss_array = np.array(h.history['loss'])
norm_loss = loss_array/loss_array[0]
fig = plt.figure()
new_plot = fig.add_subplot(111)
new_plot.plot(norm_loss, label='normalized loss')
plt.yscale('log')
plt.legend()
fig.savefig('./output/normalized_loss.png')
plt.close('all')
