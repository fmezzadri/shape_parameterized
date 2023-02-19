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
p_val = 2
c_val = 1
p = str(p_val)
c = str(c_val)

#import shape parameters
P = np.array(pd.read_csv('./C'+c+'P'+p+'/_CPs.csv.csv')).astype(np.float32)

P1 = P[:,0]      #vector containing values of first CP
P2 = P[:,1]      #vector containing values of second CP
P3 = P[:,2]      #vector containing values of third CP
P4 = P[:,3]      #vector containing values of fourth CP
P5 = P[:,4]      #vector containing values of fifth CP
P6 = P[:,5]      #vector containing values of sixth CP
P7 = P[:,6]      #vector containing values of seventh CP
P8 = P[:,7]      #vector containing values of eigth CP
P9 = P[:,8]      #vector containing values of ninth CP
P10 = P[:,9]     #vector containing values of tenth CP
P11 = P[:,10]    #vector containing values of eleventh CP
P12 = P[:,11]    #vector containing values of twelveth CP
P13 = P[:,12]    #vector containing values of thirteenth CP
P14 = P[:,13]    #vector containing values of fourteenth CP
P15 = P[:,14]    #vector containing values of fifteenth CP
P16 = P[:,15]    #vector containing values of sixteenth CP
P17 = P[:,16]    #vector containing values of seventeenth CP
P18 = P[:,17]    #vector containing values of eighteenth CP
P19 = P[:,18]    #vector containing values of nineteenth CP
P20 = P[:,19]    #vector containing values of twentieth CP

P1 = (P1-np.min(P1))/(np.max(P1)-np.min(P1)) #normalize input P1
P2 = (P2-np.min(P2))/(np.max(P2)-np.min(P2)) #normalize input P2
P3 = (P3-np.min(P3))/(np.max(P3)-np.min(P3)) #normalize input P3
P4 = (P4-np.min(P4))/(np.max(P4)-np.min(P4)) #normalize input P4
P5 = (P5-np.min(P5))/(np.max(P5)-np.min(P5)) #normalize input P5
P6 = (P6-np.min(P6))/(np.max(P6)-np.min(P6)) #normalize input P6
P7 = (P7-np.min(P7))/(np.max(P7)-np.min(P7)) #normalize input P7
P8 = (P8-np.min(P8))/(np.max(P8)-np.min(P8)) #normalize input P8
P9 = (P9-np.min(P9))/(np.max(P9)-np.min(P9)) #normalize input P9
P10 = (P10-np.min(P10))/(np.max(P10)-np.min(P10)) #normalize input P10
P11 = (P11-np.min(P11))/(np.max(P11)-np.min(P11)) #normalize input P11
P12 = (P12-np.min(P12))/(np.max(P12)-np.min(P12)) #normalize input P12
P13 = (P13-np.min(P13))/(np.max(P13)-np.min(P13)) #normalize input P13
P14 = (P14-np.min(P14))/(np.max(P14)-np.min(P14)) #normalize input P14
P15 = (P15-np.min(P15))/(np.max(P15)-np.min(P15)) #normalize input P15
P16 = (P16-np.min(P16))/(np.max(P16)-np.min(P16)) #normalize input P16
P17 = (P17-np.min(P17))/(np.max(P17)-np.min(P17)) #normalize input P17
P18 = (P18-np.min(P18))/(np.max(P18)-np.min(P18)) #normalize input P18
P19 = (P19-np.min(P19))/(np.max(P19)-np.min(P19)) #normalize input P19
P20 = (P20-np.min(P20))/(np.max(P20)-np.min(P20)) #normalize input P20

nr = 2500 #memorize number of rows in R_len #np.shape(P1)[0]
indx = np.arange(0,nr) #indices are already ordered in dataset, so indices are [0,1,...,nr-1]
#choose how many and which geometries use for training.
nds = 100 #number of training geometries
train_indx = np.arange(0,nds) #indices training geometries. Cardinality must be nds

#build training dataset
num = str(indx[train_indx[0]]+1) #+1 because naming of datasets starts at 1, not 0
dataset1 = np.array(pd.read_csv('./C'+c+'P'+p+'/_geom'+num+'.csv.csv')).astype(np.float32)
x_r = dataset1[:,0] #vector coordinates single geometry, used for dimensioning
N1 = np.size(x_r) #number of rows of a single geometry

dataset = dataset1 #initialize dataset
for i in range(1,nds):
  num = str(indx[train_indx[i]]+1)
  dataset=np.append(dataset,np.array(pd.read_csv('./C'+c+'P'+p+'/_geom'+num+'.csv.csv')).astype(np.float32))
dataset = np.reshape(dataset,(N1*nds,21))

#coordinates in parametric domain, do not change with physical shape
x_data = dataset[:,0]
y_data = dataset[:,1]
N = np.size(x_data)

#data on displacements and stresses
ux_data = dataset[:,4]
uy_data = dataset[:,5]
sigmaxx_data = dataset[:,9]
sigmayy_data = dataset[:,11]
sigmaxy_data = dataset[:,10]

#import data of inverse Jacobian
xix_data = dataset[:,16]
xiy_data = dataset[:,17]
etax_data = dataset[:,18]
etay_data = dataset[:,19]

#define shape data
A1_data = P1[indx[train_indx[0]]]*np.ones_like(x_r)
for i in range(1,nds):
  A1_data = np.append(A1_data,P1[indx[train_indx[i]]]*np.ones_like(x_r))
  
A2_data = P2[indx[train_indx[0]]]*np.ones_like(x_r)
for i in range(1,nds):
  A2_data = np.append(A2_data,P2[indx[train_indx[i]]]*np.ones_like(x_r))
  
A3_data = P3[indx[train_indx[0]]]*np.ones_like(x_r)
for i in range(1,nds):
  A3_data = np.append(A3_data,P3[indx[train_indx[i]]]*np.ones_like(x_r))
  
A4_data = P4[indx[train_indx[0]]]*np.ones_like(x_r)
for i in range(1,nds):
  A4_data = np.append(A4_data,P4[indx[train_indx[i]]]*np.ones_like(x_r))
  
A5_data = P5[indx[train_indx[0]]]*np.ones_like(x_r)
for i in range(1,nds):
  A5_data = np.append(A5_data,P5[indx[train_indx[i]]]*np.ones_like(x_r))
  
A6_data = P6[indx[train_indx[0]]]*np.ones_like(x_r)
for i in range(1,nds):
  A6_data = np.append(A6_data,P6[indx[train_indx[i]]]*np.ones_like(x_r))
  
A7_data = P7[indx[train_indx[0]]]*np.ones_like(x_r)
for i in range(1,nds):
  A7_data = np.append(A7_data,P7[indx[train_indx[i]]]*np.ones_like(x_r))
  
A8_data = P8[indx[train_indx[0]]]*np.ones_like(x_r)
for i in range(1,nds):
  A8_data = np.append(A8_data,P8[indx[train_indx[i]]]*np.ones_like(x_r))
  
A9_data = P9[indx[train_indx[0]]]*np.ones_like(x_r)
for i in range(1,nds):
  A9_data = np.append(A9_data,P9[indx[train_indx[i]]]*np.ones_like(x_r))
  
A10_data = P10[indx[train_indx[0]]]*np.ones_like(x_r)
for i in range(1,nds):
  A10_data = np.append(A10_data,P10[indx[train_indx[i]]]*np.ones_like(x_r))
  
A11_data = P11[indx[train_indx[0]]]*np.ones_like(x_r)
for i in range(1,nds):
  A11_data = np.append(A11_data,P11[indx[train_indx[i]]]*np.ones_like(x_r))
  
A12_data = P12[indx[train_indx[0]]]*np.ones_like(x_r)
for i in range(1,nds):
  A12_data = np.append(A12_data,P12[indx[train_indx[i]]]*np.ones_like(x_r))
  
A13_data = P13[indx[train_indx[0]]]*np.ones_like(x_r)
for i in range(1,nds):
  A13_data = np.append(A13_data,P13[indx[train_indx[i]]]*np.ones_like(x_r))
  
A14_data = P14[indx[train_indx[0]]]*np.ones_like(x_r)
for i in range(1,nds):
  A14_data = np.append(A14_data,P14[indx[train_indx[i]]]*np.ones_like(x_r))
  
A15_data = P15[indx[train_indx[0]]]*np.ones_like(x_r)
for i in range(1,nds):
  A15_data = np.append(A15_data,P15[indx[train_indx[i]]]*np.ones_like(x_r))
  
A16_data = P16[indx[train_indx[0]]]*np.ones_like(x_r)
for i in range(1,nds):
  A16_data = np.append(A16_data,P16[indx[train_indx[i]]]*np.ones_like(x_r))
  
A17_data = P17[indx[train_indx[0]]]*np.ones_like(x_r)
for i in range(1,nds):
  A17_data = np.append(A17_data,P17[indx[train_indx[i]]]*np.ones_like(x_r))
  
A18_data = P18[indx[train_indx[0]]]*np.ones_like(x_r)
for i in range(1,nds):
  A18_data = np.append(A18_data,P18[indx[train_indx[i]]]*np.ones_like(x_r))
  
A19_data = P19[indx[train_indx[0]]]*np.ones_like(x_r)
for i in range(1,nds):
  A19_data = np.append(A19_data,P19[indx[train_indx[i]]]*np.ones_like(x_r))
  
A20_data = P20[indx[train_indx[0]]]*np.ones_like(x_r)
for i in range(1,nds):
  A20_data = np.append(A20_data,P20[indx[train_indx[i]]]*np.ones_like(x_r))
  
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
A9 = sn.Variable("A9", dtype=dtype)
A10 = sn.Variable("A10", dtype=dtype)
A11 = sn.Variable("A11", dtype=dtype)
A12 = sn.Variable("A12", dtype=dtype)
A13 = sn.Variable("A13", dtype=dtype)
A14 = sn.Variable("A14", dtype=dtype)
A15 = sn.Variable("A15", dtype=dtype)
A16 = sn.Variable("A16", dtype=dtype)
A17 = sn.Variable("A17", dtype=dtype)
A18 = sn.Variable("A18", dtype=dtype)
A19 = sn.Variable("A19", dtype=dtype)
A20 = sn.Variable("A20", dtype=dtype)
xix = sn.Variable("xix", dtype=dtype)
xiy = sn.Variable("xiy", dtype=dtype)
etax = sn.Variable("etax", dtype=dtype)
etay = sn.Variable("etay", dtype=dtype)

#define the functionals of the neural network
Uxy = sn.Functional("Uxy", [x, y, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20], 5*[20], 'tanh')
Vxy = sn.Functional("Vxy", [x, y, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20], 5*[20], 'tanh')
Sxx = sn.Functional("Sxx", [x, y, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20], 5*[20], 'tanh')
Syy = sn.Functional("Syy", [x, y, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20], 5*[20], 'tanh')
Sxy = sn.Functional("Sxy", [x, y, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20], 5*[20], 'tanh')

#compute elasticity parameters
C11 = E/(1-nu**2)
C12 = nu*E/(1-nu**2)
C33 = E*(1-nu)/(2*(1-nu**2))

#compute deformations (using chain rule) for residual computations
Exx = sn.diff(Uxy, x)*xix + sn.diff(Uxy, y)*etax
Eyy = sn.diff(Vxy, x)*xiy + sn.diff(Vxy, y)*etay
Gxy = sn.diff(Uxy, x)*xiy + sn.diff(Uxy, y)*etay + sn.diff(Vxy, x)*xix + sn.diff(Vxy, y)*etax

#assign data
Du = sn.Data(Uxy)
Dv = sn.Data(Vxy)
Ds1 = sn.Data(Sxx)
Ds2 = sn.Data(Sxy)
Ds3 = sn.Data(Syy)

targets = [Du, Dv, Ds1, Ds2, Ds3]

#define scimodel
m = sn.SciModel([x, y, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, xix, xiy, etax, etay], targets)

#train model
h = m.train([x_data, y_data, A1_data, A2_data, A3_data, A4_data, A5_data, A6_data, A7_data, A8_data, A9_data, A10_data, A11_data, A12_data, A13_data, A14_data, A15_data, A16_data, A17_data, A18_data, A19_data, A20_data, xix_data, xiy_data, etax_data, etay_data], [ux_data, uy_data, sigmaxx_data, sigmaxy_data, sigmayy_data], batch_size=100, epochs=1200,  learning_rate = ([0, 300, 700, 1100], [0.001, 0.0005, 0.0002, 0.0001]), log_parameters={}, adaptive_weights={'method':'NTK', 'freq': 100}, verbose = 2)


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
