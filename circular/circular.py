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

#setting of the experiment
flname = 'datasetCIRC' #name of the folder containing the data
mode = 'PINN' #choose between PINN and NN
e_tot = 500 #total number of epochs

#elasticity parameters
E = 1000.
nu = 0.3

#Define training lengths
L_dat = [4.241, 5.4618, 6.2651, 7.0201, 7.5502, 8.0]
nL_data = np.size(L_dat)

#import datasets
dataset = []

for i in range(nL_data):
  num = str(L_dat[i])
  dataset = np.append(dataset, np.array(pd.read_csv("./" + flname + "/dataOmega" + num + ".csv")).astype(np.float32))
  if i == 0:
    x_r = np.reshape(dataset, (10000*1,19))[:,0] #store vector of coordinates

dataset = np.reshape(dataset, (10000*nL_data,19))

#coordinates in parametric domain, do not change with physical shape
x_data = dataset[:,0]
y_data = dataset[:,1]
N = np.size(x_data)

#displacement and stess data
ux_data = dataset[:,4]
uy_data = dataset[:,5]
sigmaxx_data = dataset[:,9]
sigmayy_data = dataset[:,10]
sigmaxy_data = dataset[:,11]

#import data of inverse Jacobian
xix_data = dataset[:,14]
xiy_data = dataset[:,15]
etax_data = dataset[:,16]
etay_data = dataset[:,17]

#build L data
L_data = []
for i in range(nL_data):
  L_data = np.append(L_data,L_dat[i]*np.ones_like(x_r))

########################
# Neural Network Setup
########################
dtype='float32'

#define the variables of the neural network
x = sn.Variable("x", dtype=dtype)
y = sn.Variable("y", dtype=dtype)
L = sn.Variable("L", dtype=dtype)
xix = sn.Variable("xix", dtype=dtype)
xiy = sn.Variable("xiy", dtype=dtype)
etax = sn.Variable("etax", dtype=dtype)
etay = sn.Variable("etay", dtype=dtype)

#define the functionals of the neural network
Uxy = sn.Functional("Uxy", [x, y, L], 5*[20], 'tanh')
Vxy = sn.Functional("Vxy", [x, y, L], 5*[20], 'tanh')
Sxx = sn.Functional("Sxx", [x, y, L], 5*[20], 'tanh')
Syy = sn.Functional("Syy", [x, y, L], 5*[20], 'tanh')
Sxy = sn.Functional("Sxy", [x, y, L], 5*[20], 'tanh')

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

#uncomment next lines to use momentum equations as well
#Sxxx = sn.diff(Sxx, x)*xix + sn.diff(Sxx, y)*etax
#Sxyx = sn.diff(Sxy, x)*xix + sn.diff(Sxy, y)*etax
#Sxyy = sn.diff(Sxy, x)*xiy + sn.diff(Sxy, y)*etay
#Syyy = sn.diff(Syy, x)*xiy + sn.diff(Syy, y)*etay
#Lx = Sxxx + Sxyy
#Ly = Sxyx + Syyy


#assign data
Du = sn.Data(Uxy)
Dv = sn.Data(Vxy)
Ds1 = sn.Data(Sxx)
Ds2 = sn.Data(Sxy)
Ds3 = sn.Data(Syy)

if mode in ['PINN']:
  targets = [Du, Dv, Ds1, Ds2, Ds3, c1, c2, c3]
  #targets = [Du, Dv, Ds1, Ds2, Ds3, c1, c2, c3, Lx, Ly] #with momentum
else:
  targets = [Du, Dv, Ds1, Ds2, Ds3]

#set up Sci Model
m = sn.SciModel([x, y, L, xix, xiy, etax, etay], targets)

#residual data (zero residual)
c1_data = np.zeros_like(y_data)
c2_data = np.zeros_like(y_data)
c3_data = np.zeros_like(y_data)

#uncomment next lines to use momentum equations as well
#Lx_data = np.zeros_like(y_data)
#Ly_data = np.zeros_like(y_data)

#train network
if mode in ['PINN']:
  h = m.train([x_data, y_data, L_data, xix_data, xiy_data, etax_data, etay_data], [ux_data, uy_data, sigmaxx_data, sigmaxy_data, sigmayy_data, c1_data, c2_data, c3_data],batch_size=100, epochs=e_tot, log_parameters={},adaptive_weights={'method':'NTK', 'freq': 100}, verbose = 2)
  #h = m.train([x_data, y_data, L_data, xix_data, xiy_data, etax_data, etay_data], [ux_data, uy_data, sigmaxx_data, sigmaxy_data, sigmayy_data, c1_data, c2_data, c3_data, Lx_data, Ly_data],batch_size=100, epochs=e_tot, log_parameters={}, adaptive_weights={'method':'NTK', 'freq': 100}, verbose = 2) #with momentum equation
else:
  h = m.train([x_data, y_data, L_data, xix_data, xiy_data, etax_data, etay_data], [ux_data, uy_data, sigmaxx_data, sigmaxy_data, sigmayy_data],
            batch_size=100, epochs=e_tot, log_parameters={},
            adaptive_weights={'method':'NTK', 'freq': 100}, verbose = 2)

#########################################################################
##########                 END OF PINN MODEL                  ###########
##########     GENEREATION AND PRINTING OF RESULTS BEGINS     ###########
#########################################################################

#save the weights of the trained PINN
m.save_weights('./output/weights_PINN.ckpt')
#copy output (if it was created) to output folder
#shutil.copy('./terminal_output.txt', './output/terminal_output.txt')

#save the plot of the total loss function
plt.plot(h.history['loss'], label='loss')
plt.yscale('log')
plt.legend()
plt.savefig('./output/loss.png')
plt.close()

#plot error at all points
lmax = 249

#allocate vectors to store errors, etc...
L_vec = []
err_ux = []
err_uy = []
err_sxx = []
err_syy = []
err_sxy = []
err_ux_norm = []
err_uy_norm = []
err_sxx_norm = []
err_syy_norm = []
err_sxy_norm = []
ux_mean_vec = []
uy_mean_vec = []
sxx_mean_vec = []
syy_mean_vec = []
sxy_mean_vec = []

#import a dataset to store the location of the collocation points
num = str(round(4 + 4./lmax,4))
dataset_val = np.array(pd.read_csv('./' + flname + '/dataOmega'+num+'.csv')).astype(np.float32)
xi_val = dataset_val[:,0]
eta_val = dataset_val[:,1]
Nv = np.size(xi_val)

#loop to compute errors as the inner radius varies
for i in range(lmax):
  num = str(round(4 + 4./lmax*i,4))
  dataset_val = np.array(pd.read_csv('./' + flname + '/dataOmega'+num+'.csv')).astype(np.float32)

  #import real solution
  ux_star = dataset_val[:,4]
  uy_star = dataset_val[:,5]
  sxx_star = dataset_val[:,9]
  syy_star = dataset_val[:,10]
  sxy_star = dataset_val[:,11]

  #compute and store inner radius of the considered shape
  L_i = 4 + 4./lmax*i
  L_vec.append(L_i)
  L_val = L_i*np.ones_like(xi_val)

  #use trained neural network to make predictions
  ux_end = Uxy.eval([xi_val, eta_val, L_val])
  uy_end = Vxy.eval([xi_val, eta_val, L_val])
  sxx_end = Sxx.eval([xi_val, eta_val, L_val])
  syy_end = Syy.eval([xi_val, eta_val, L_val])
  sxy_end = Sxy.eval([xi_val, eta_val, L_val])

  #compute errors
  err_ux.append((np.sum((ux_end-ux_star)**2))/Nv)
  err_uy.append((np.sum((uy_end-uy_star)**2))/Nv)
  err_sxx.append((np.sum((sxx_end-sxx_star)**2))/Nv)
  err_syy.append((np.sum((syy_end-syy_star)**2))/Nv)
  err_sxy.append((np.sum((sxy_end-sxy_star)**2))/Nv)

  #compute mean solution fields for normalization
  ux_mean = np.sum(ux_star**2)/Nv
  uy_mean = np.sum(uy_star**2)/Nv
  sxx_mean = np.sum(sxx_star**2)/Nv
  syy_mean = np.sum(syy_star**2)/Nv
  sxy_mean = np.sum(sxy_star**2)/Nv
  ux_mean_vec.append(ux_mean)
  uy_mean_vec.append(uy_mean)
  sxx_mean_vec.append(sxx_mean)
  syy_mean_vec.append(syy_mean)
  sxy_mean_vec.append(sxy_mean)

  #compute normalized errors
  err_ux_norm.append((np.sum((ux_end-ux_star)**2))/Nv/ux_mean)
  err_uy_norm.append((np.sum((uy_end-uy_star)**2))/Nv/uy_mean)
  err_sxx_norm.append((np.sum((sxx_end-sxx_star)**2))/Nv/sxx_mean)
  err_syy_norm.append((np.sum((syy_end-syy_star)**2))/Nv/syy_mean)
  err_sxy_norm.append((np.sum((sxy_end-sxy_star)**2))/Nv/sxy_mean)

#save error vectors to output directory
with open('./output/err_ux.txt', 'w') as f:
    print(*err_ux, file=f)
with open('./output/err_uy.txt', 'w') as f:
    print(*err_uy, file=f)
with open('./output/err_sxx.txt', 'w') as f:
    print(*err_sxx, file=f)
with open('./output/err_syy.txt', 'w') as f:
    print(*err_syy, file=f)
with open('./output/err_sxy.txt', 'w') as f:
    print(*err_sxy, file=f)

with open('./output/err_ux_norm.txt', 'w') as f:
    print(*err_ux_norm, file=f)
with open('./output/err_uy_norm.txt', 'w') as f:
    print(*err_uy_norm, file=f)
with open('./output/err_sxx_norm.txt', 'w') as f:
    print(*err_sxx_norm, file=f)
with open('./output/err_syy_norm.txt', 'w') as f:
    print(*err_syy_norm, file=f)
with open('./output/err_sxy_norm.txt', 'w') as f:
    print(*err_sxy_norm, file=f)

#plot errors and save to output directory
plt.plot(L_vec, err_ux)
plt.yscale('log')
plt.plot(L_vec, err_uy)
plt.yscale('log')
plt.plot(L_vec, err_sxx)
plt.yscale('log')
plt.plot(L_vec, err_syy)
plt.yscale('log')
plt.plot(L_vec, err_sxy)
plt.yscale('log')
plt.legend(['ux', 'uy', 'sxx', 'syy', 'sxy'])
plt.savefig('./output/error.png', dpi=1200)
plt.close()

plt.plot(L_vec, err_ux_norm)
plt.yscale('log')
plt.plot(L_vec, err_uy_norm)
plt.yscale('log')
plt.plot(L_vec, err_sxx_norm)
plt.yscale('log')
plt.plot(L_vec, err_syy)
plt.yscale('log')
plt.plot(L_vec, err_sxy_norm)
plt.yscale('log')
plt.legend(['ux', 'uy', 'sxx', 'syy', 'sxy'])
plt.savefig('./output/error_norm.png', dpi=1200)
