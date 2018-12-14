import numpy as np
from torch import tensor
from doa_math import tensor_angle

class ToleranceScore:
  def __init__(self,thresholds,doa_classes):
    size = len(thresholds)
    self.CC = np.zeros(size)
    self.CX = np.zeros(size)
    self.XC = np.zeros(size)
    self.XX = np.zeros(size)
    self.thresholds = thresholds
    self.doa_classes = doa_classes
    
  def update(self,Yhat,Y):
    cc,cx,xc,xx = angular_errors(Yhat,Y,self.thresholds,self.doa_classes)
    CC += cc
    CX += cx
    XC += xc
    XX += xx
    
  def __repr__(self):
    return '\n'.join(["{}={}".format(p,self.__dict__[p]) for p in self.__dict__])

def ratios_less(X,thresholds):
  return np.array([sum(X <= t) / float(len(X)) for t in thresholds])

def angular_errors(Yhat,Y,thresholds,doa_classes):
  Yhat = tensor(Yhat)
  Yhat_c = tensor(snap_all(Yhat,doa_classes))
  Y = tensor(Y)
  Y_c = tensor(snap_all(Y,doa_classes))
  cc = ratios_less(tensor_angle(Yhat_c,Y_c),thresholds)
  cx = ratios_less(tensor_angle(Yhat_c,Y),thresholds)
  xc = ratios_less(tensor_angle(Yhat,Y_c),thresholds)
  xx = ratios_less(tensor_angle(Yhat,Y),thresholds)
  return cc,cx,xc,xx
  
class SNRTestDatasets:
  def __init__(self):
    

class SNRCurve:
  def __init__(self,):
    tolerance_scores = [ToleranceScore() for i in]
    
