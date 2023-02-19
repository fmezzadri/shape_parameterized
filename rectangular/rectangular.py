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
flname = 'length' #name of the folder containing the data
mode = 'PINN' #choose between PINN and NN
e_tot = 500 #total number of epochs

#elasticity parameters
mu = 0.5
lmbd = 1.

#Define training lengths
L_dat = [2,3,5,10]
nL_data = np.size(L_dat)

#import datasets
dataset = []

for i in range(nL_data):
  num = str(L_dat[i])
  dataset = np.append(dataset, np.array(pd.read_csv("./" + flname + "/dataOmega_L" + num + ".csv")).astype(np.float32))
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
E = 2*(1+lmbd/(2*(lmbd+mu)))*mu
nu = lmbd/(2*(lmbd+mu))

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

if mode in ['PINN']:
  targets = [Du, Dv, Ds1, Ds2, Ds3, c1, c2, c3]
else:
  targets = [Du, Dv, Ds1, Ds2, Ds3]

#set up Sci Model
m = sn.SciModel([x, y, L, xix, xiy, etax, etay], targets)

#residual data (zero residual)
c1_data = np.zeros_like(y_data)
c2_data = np.zeros_like(y_data)
c3_data = np.zeros_like(y_data)

#train network
if mode in ['PINN']:
  h = m.train([x_data, y_data, L_data, xix_data, xiy_data, etax_data, etay_data], [ux_data, uy_data, sigmaxx_data, sigmaxy_data, sigmayy_data, c1_data, c2_data, c3_data],batch_size=100, epochs=e_tot, log_parameters={},adaptive_weights={'method':'NTK', 'freq': 100}, verbose = 2)
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

################
#compare with analytical solution
################

#define parameters of the problem to compute analytical solutions
P = 1.0
L = 3.0
c = 0.75
t = 1.0
I = 2*t*c**3/3.0
E = 2*(1+lmbd/(2*(lmbd+mu)))*mu
nu = lmbd/(2*(lmbd+mu))
G = E/(2*(1+nu))

#x component analytical solution, ux
def dispx(x,y,L):
    return -P*(x**2-L**2)*y/(2*E*I) - nu*P*y*(y**2-c**2)/(6*E*I) + P*y*(y**2-c**2)/(6*G*I)

#y component analytical solution, uy
def dispy(x,y,L):
    return nu*P*x*y**2/(2*E*I) + P*(x**3-L**3)/(6*E*I) - (P*L**2/(2*E*I)+nu*P*c**2/(6*E*I)+P*c**2/(3*G*I))*(x-L)

#sigma_xx
def stressxx(x,y):
    return -1.5*P*x*y/c**3

#sigma_yy
def stressyy(x,y):
    return 0.0

#sigma_xy
def stressxy(x,y):
    return -3.0/4.0*P/c*(1.0-y**2/c**2)

#import a dataset to store the location of the collocation points
dataset1 = np.array(pd.read_csv("./length/dataOmega_L2.csv")).astype(np.float32)
xi_val = dataset1[:,0]
eta_val = dataset1[:,1]
x_val2 = dataset1[:,2]
y_val2 = dataset1[:,3]
Nv = np.size(x_val2)

#choose the interval where the accuracy of the network will be evaluated
delta_L = 0.1
L_min = 1.9
L_max = 10.1
Nexp = int((L_max-L_min)/delta_L+1)

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

#loop to compute errors in the chosen range of lengths
for i in range(1,Nexp):

  L_i = L_min + (i-1)*(L_max-L_min)/Nexp
  #modify physical coordinates by using linearity of Jacobian
  x_val = x_val2/2.0*L_i
  y_val = y_val2

  #store considered values of length
  L_vec.append(L_i)
  L_val = L_i*np.ones_like(x_val)

  #use trained neural network to make predictions
  ux_end = Uxy.eval([xi_val, eta_val, L_val])
  uy_end = Vxy.eval([xi_val, eta_val, L_val])
  sxx_end = Sxx.eval([xi_val, eta_val, L_val])
  syy_end = Syy.eval([xi_val, eta_val, L_val])
  sxy_end = Sxy.eval([xi_val, eta_val, L_val])

  #compute analytical solution at the same length
  ux_star = dispx(x_val, y_val, L_i)
  uy_star = dispy(x_val, y_val, L_i)
  sxx_star = stressxx(x_val, y_val)
  syy_star = stressyy(x_val, y_val)
  sxy_star = stressxy(x_val, y_val)
  
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
  
#plot absolute errors and save to output directory
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
with open('./output/mean_ux.txt', 'w') as f:
    print(*ux_mean_vec, file=f)
with open('./output/mean_uy.txt', 'w') as f:
    print(*uy_mean_vec, file=f)
with open('./output/mean_sxx.txt', 'w') as f:
    print(*sxx_mean_vec, file=f)
with open('./output/mean_syy.txt', 'w') as f:
    print(*syy_mean_vec, file=f)
with open('./output/mean_sxy.txt', 'w') as f:
    print(*sxy_mean_vec, file=f)
with open('./output/loss.txt', 'w') as f:
    print(*loss_array, file=f)
