import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

class DoaClass():
  def __init__(self, elevation, azimuth):
    self.elevation = elevation
    self.azimuth = azimuth
    self.inclination = (np.pi/2) - self.elevation
    self.x = np.sin(self.inclination)*np.cos(self.azimuth)
    self.y = np.sin(self.inclination)*np.sin(self.azimuth)
    self.z = np.cos(self.inclination)

class DoaClasses():
  def __init__(self, doa_grid_resolution = np.pi/18):
    self.classes = self.generate_direction_classes(doa_grid_resolution) 

  def generate_direction_classes(self, resolution):
    direction_classes = []
    num_elevations = int(np.pi//resolution)
    for i in range(num_elevations):
      elevation = (-np.pi/2) + np.pi*i/num_elevations
      num_azimuths = int(2*np.pi*np.cos(elevation)//resolution)
      for j in range(num_azimuths+1):
        azimuth = -np.pi + 2*np.pi*j/(num_azimuths + 1)
        direction_classes.append(DoaClass(elevation, azimuth))

    return direction_classes

  def plot_classes(self):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = []
    ys = []
    zs = []
    for doa_class in self.classes:
      xs.append(doa_class.x)
      ys.append(doa_class.y)
      zs.append(doa_class.z)
    zeros = [0]*len(self.classes)
    ax.scatter(xs,ys,zs,s=2)
  #  ax.quiver(zeros,zeros,zeros,xs,ys,zs,arrow_length_ratio=0.01)
  #  ax.set_xlim3d(-1, 1)
  #  ax.set_ylim3d(-1,1)
  #  ax.set_zlim3d(-1,1)
    plt.show()

def tensor_angle(a, b):
  inner_product = (a * b).sum(dim=1)
  a_norm = a.pow(2).sum(dim=1).pow(0.5)
  b_norm = b.pow(2).sum(dim=1).pow(0.5)
  cos = inner_product / (a_norm * b_norm)
  angle = torch.acos(cos)
  angle[torch.isnan(angle)] = 0
  return angle
    
def to_cartesian(x,doa_classes):
  assert x < len(self.classes)
  doa_class = self.classes[x]
  return doa_class.get_xyz_vector()

def to_class(xyz,doa_classes):
  max_dot_product = -1
  class_index = -1
  for i, doa_class in enumerate(self.classes):
    dp = np.dot(xyz, doa_class.get_xyz_vector())
    if dp > max_dot_product:
      max_dot_product = dp
      class_index = i
  assert class_index > -1
  return class_index
    
def snap(x,doa_classes):
  return to_cartesian(to_class(x,doa_classes),doa_classes)
  
def snap_all(X,doa_classes):
  return [snap(x,doa_classes) for x in X]

