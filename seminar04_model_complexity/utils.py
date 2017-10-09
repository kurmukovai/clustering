#
# coding: utf-8
# Auxilary functions
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
from math import pi

########## Input generation
def input_generator(sample_size, minmax_values=None, x_dim=2):
  x = np.random.rand(sample_size, x_dim) 

  # set bounds
  if not minmax_values == None:
    for j in range(x_dim):
      x[:, j] = x[:, j] * (minmax_values[j][1] - minmax_values[j][0]) + minmax_values[j][0]
  return x
  
########## Output generation
def output_generator(x, function, noise_scale_factor = 0):
  if function == 'branin':
    const = [5.1 / 4.0 / pi**2, 5.0 / pi, -6.0, 10.0 * (1.0 - 1.0 / 8.0 / pi), 10.0]
    y = (x[:, 1] - const[0] * x[:, 0]**2 + const[1] * x[:, 0] + const[2])**2 + const[3] * np.cos(x[:, 0]) + const[4]
  elif function == 'michalewicz':
    y = np.zeros((x.shape[0],1))[:, 0]
    for i in range(x.shape[1]):
      y = y + np.sin(x[:, i]) * np.sin(x[:, i]**2 / pi)
  elif function == 'golubev':
    intermediate = 5 * np.exp(-4 * np.sum((x - 0.25)**2, 1)) - \
    7 * np.exp(-40 * np.sum(x**2, 1)) - \
    3 * np.exp(-10 * (x[:, 0] + 0.7)**2) + np.sum(x, 1)
    y = np.tanh(0.3 * intermediate)
  elif function == 'gSobol':
    a = [4.5, 4.5, 1, 0, 1, 9];
    y = np.ones((x.shape[0],1))[:, 0]
    for i in range(x.shape[1]):
      y = y * (np.abs(4 * x[:, i] - 2) + a[i]) / (1 + a[i])
  return y + noise_scale_factor * np.random.randn(x.shape[0], 1)[:, 0]

########## Errors
def calculate_rrms(values, values_predicted, train_values):
  values = vectorize(values)  
  values_predicted = vectorize(values_predicted)
  train_values = vectorize(train_values)
  
  mean_values = vectorize(np.tile(np.mean(train_values, axis = 0), (values.shape[0], 1)))
  const_error = (np.mean(np.mean((values - mean_values)**2)))**0.5
  residuals = np.abs(values - values_predicted)
  RRMS = (np.mean(np.mean(residuals**2)))**0.5 / const_error
  return RRMS
  
def vectorize(matrix):
  matrix = np.array(matrix)
  if len(matrix.shape) == 2: 
    return matrix[:, 0]
  else:
    return matrix
########## 3d plots
def plot_surface(calc_method, scatter_x=None, scatter_y=None, minmax_values = [[-1, 1], [-1, 1]], \
      view_angles=None, zlim = None, title=None, filename='tmp', prefix='tutorial_approx_'):
      
  fig = plt.figure(figsize=(10.0,7.0))
  ax = Axes3D(fig)
  
  if not view_angles == None:
    ax.view_init(view_angles[0], view_angles[1])
  # training sample scatter plot  
  if not np.any(scatter_x) == None:# & not scatter_y=None:
    ax.scatter3D(scatter_x[:, 0], scatter_x[:, 1], scatter_y, alpha = 1.0, c ='r', marker='o', linewidth = 1, s = 50)
  
  # generate mesh  
  grid_x1, grid_x2 = mesh_generator_aux([40, 40], minmax_values) 
  reshape_sizes = [grid_x1.shape]
  grid_x = np.hstack([grid_x1.reshape(np.prod(reshape_sizes), 1), grid_x2.reshape(np.prod(reshape_sizes), 1)])
  grid_values = np.array(calc_method(grid_x))
  # plot surface  
  ax.plot_surface(grid_x1, grid_x2, grid_values.reshape(reshape_sizes[0]), rstride = 1, cstride = 1, cmap = cm.jet,
      linewidth = 0.1, alpha = 0.6, antialiased = False)
  
  # appearance   
  if not title == None:
    plt.suptitle(title, fontsize = 24)
  ax.set_xlabel('X1', fontsize = 18)  
  ax.set_ylabel('X2', fontsize = 18)
  if not zlim == None:
    ax.set_zlim3d(zlim[0], zlim[1])

  plot_filename = prefix + filename + '.png'
  plt.savefig(plot_filename, format='png')
  print('Plot is saved to %s' % os.path.join(os.getcwd(), plot_filename))
  plt.show()
  
def show_plots():
  if not os.environ.has_key('SUPPRESS_SHOW_PLOTS'):
    print('\nClose all plot windows to continue script. \n')
    plt.show()
  plt.close()
  
def mesh_generator_aux(points_number, minmax_values):
  x1 = np.linspace(minmax_values[0][0],minmax_values[0][1],points_number[0])
  x2 = np.linspace(minmax_values[1][0],minmax_values[1][1],points_number[1])
  grid_x1, grid_x2 = np.meshgrid(x1, x2)
  return grid_x1, grid_x2
    
########## Airfoil-pressure plots
def airfoil_plotting(test_x, test_y, test_prediction):
  print('Visualizing airfoils and pressure distributions (original and approximated) on test set')

  fig = plt.figure(figsize=(15, 9))
  for idx in range(test_x.shape[0]):
    ax = fig.add_subplot(2, 2, 1 + idx * 2)
    plot_afl(ax, list(test_x[idx, :]))
    ax = fig.add_subplot(2, 2, 2 + idx * 2)
    plot_pressure(ax, list(test_y[idx, :]), list(test_prediction[idx, :]))
   
  plt.suptitle('2 airfoil profiles and pressure distribution for them', fontsize = 24)
  plot_filename = 'tutorial_approx_airfoil_pressure.png'
  plt.savefig(plot_filename, format='png')
  print('Plot is saved to %s' % os.path.join(os.getcwd(), plot_filename))
  plt.show()

def plotting_counts(mode):
  counts = [1,0.99,0.975,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.075,
       0.05,0.03,0.02,0.01,0.005,0.0025,0.001,0,0.001,0.0025,0.005,0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,
       0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.975,0.99,1];
  if mode == 'pressure':
    counts.pop(29)
  else:
    counts.append(1)
  return counts
  
def plot_afl(ax, afl):
  # add redundant points & connect line ends
  afl.insert(29, 0)
  afl.insert(58, -afl[0])
  afl.insert(59, afl[0])
  # read counts & plot airfoil
  ax.plot(plotting_counts('airfoil'), afl)
  # set some plot properties
  ax.set_xlabel('x', fontsize = 16)
  ax.set_ylabel('y', fontsize = 16)
  ax.set_xlim(-0.02, 1.02)

def plot_pressure(ax, pres, pres_predict):
  # connect line ends
  pres.insert(57, pres[0])
  pres_predict.insert(57, pres_predict[0])
  # plot true & predicted pressure distributions
  ax.plot(plotting_counts('pressure'), pres, 'b-', label='Original pressure distribution')
  ax.plot(plotting_counts('pressure'), pres_predict, 'r-', label='Predicted pressure distribution')
  # set some plot properties
  ax.legend(loc='best', prop=dict(size='large'))
  ax.set_ylabel('Pressure coefficient', fontsize=18)
  ax.set_xlabel('x', fontsize = 16)
  ax.set_xlim(-0.02, 1.02)

def calculate_errors(values, values_predicted, train_values):
  mean_values = np.tile(np.mean(train_values, axis = 0), (values.shape[0], 1))
  const_error = (np.mean(np.mean((values - mean_values)**2)))**0.5
  
  residuals = np.abs(values - values_predicted)
  RMS = (np.mean(np.mean(residuals**2)))**0.5 
  MAE = np.mean(np.mean(residuals))
  print(' - mean absolute error (MAE) is ' + str(MAE) + ';')
  print(' - root-mean-square error (RMS) is ' + str(RMS) + ';')
  print(' - relative root-mean-square error (RRMS) is ' + str(RMS / const_error) + ';\n')

def print_stat(model, train_x, train_y, test_x, test_y):
  print(' - train sample: ' + str(calculate_rrms(train_y, model.calc(train_x), train_y)))
  print(' - test sample:  ' + str(calculate_rrms(test_y, model.calc(test_x), train_y)))
  print(' - maximal absolute value of derivative: ' + str(max_derivative(model, test_x)) + '\n')
  

def max_derivative(model, X):
  import numpy as np
  derivatives = []
  for x in X:
    derivatives.append(model.grad(x)[0])
  return np.max(np.abs(np.array(derivatives)))
