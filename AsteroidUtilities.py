from math import sin,cos,pi
import numpy as np
import numpy.matrixlib as nm


mtxlist = ( None, None
### 2-term Fourier fit:       a0        a1     b1      a2      b2
          , nm.matrix( [ [-15.94,   -7.522, 48.97, -5.918, -3.235 ]
                       , [  2.484, -22.72,  -9.989, 2.358, -4.198 ]
                       , [ 34.93,   -2.168, 10.02, -1.838, -0.6224]
                       ] ).T
### 3-term Fourier fit:        a0      a1       b1       a2    b2       a3       b3
          , nm.matrix( [ [  19.04, -37.98,   -0.4392, -0.4125, -8.412,  1.689,  0.5891 ]
                       , [  13.87,   4.463, -19.73,    4.672,   1.369, -0.7784, 0.889 ]
                       , [  -4.907, -8.234,   0.601,  -0.0403, -1.91,   0.442,   0.1858]
                       ] ).T
          , )


def buildvec(theta, order):
  lclOrder = order - 1
  vec = [1.0, cos(theta), sin(theta)]
  while lclOrder > 0:
    ###      cos(nT+T) =                    ,   sin(nT+T) =
    ###      cos(nT)*cos(T) - sin(nT)*sin(T),   cos(nT)*sin(T) + sin(nT)*cos(T)
    vec += [ vec[-2]*vec[1] - vec[-1]*vec[2],   vec[-1]*vec[1] + vec[-2]*vec[2] ]
    lclOrder -= 1
  return np.array(vec)


def MagFit(theta,order=2):
  return (buildvec(theta,order) * mtxlist[order]).getA1()


def getVertsTris(filepath):
  xdata = []
  ydata = []
  zdata = []
  triangles = []
  for line in open(filepath,"rb"):
    toks = line.split()
    if len(toks):
      if toks[0] == 'v':
        xdata.append(toks[1])
        ydata.append(toks[2])
        zdata.append(toks[3])
      elif toks[0] == 'f':
        triangles.append(tuple([int(tok.split("//")[0])-1 for tok in toks[1:4]]))

  ### Convert vertex tokens to NumPy arrays, return them plus triangle indices
  return ( np.array(xdata, dtype=np.float)
         , np.array(ydata, dtype=np.float)
         , np.array(zdata, dtype=np.float)
         , triangles
         , )


### Test code
if __name__=="__main__":
  ### testing
  def OldMagFitOrder3(theta):

    B = np.zeros(3)

    ax0 =  19.04
    ax1 = -37.98
    bx1 = -0.4392
    ax2 = -0.4125
    bx2 = -8.412
    ax3 =  1.689
    bx3 =  0.5891

    ay0 =  13.87
    ay1 =  4.463
    by1 = -19.73
    ay2 =  4.672
    by2 =  1.369
    ay3 = -0.7784
    by3 =  0.889

    az0 = -4.907
    az1 = -8.234
    bz1 =  0.601
    az2 = -0.0403
    bz2 = -1.91
    az3 =  0.442
    bz3 =  0.1858

    B[0] = ax0 + ax1*cos(theta) + bx1*sin(theta) + ax2*cos(2*theta) + bx2*sin(2*theta) + ax3*cos(3*theta) + bx3*sin(3*theta)
    B[1] = ay0 + ay1*cos(theta) + by1*sin(theta) + ay2*cos(2*theta) + by2*sin(2*theta) + ay3*cos(3*theta) + by3*sin(3*theta)
    B[2] = az0 + az1*cos(theta) + bz1*sin(theta) + az2*cos(2*theta) + bz2*sin(2*theta) + az3*cos(3*theta) + bz3*sin(3*theta)

    return B


  def OldMagFitOrder2(theta):

    B = np.zeros(3)

    ax0 = -15.94
    ax1 = -7.522
    bx1 =  48.97
    ax2 = -5.918
    bx2 = -3.235

    ay0 =  2.484
    ay1 = -22.72
    by1 = -9.989
    ay2 =  2.358
    by2 = -4.198

    az0 =  34.93
    az1 = -2.168
    bz1 =  10.02
    az2 = -1.838
    bz2 = -0.6224

    B[0] = ax0 + ax1*cos(theta) + bx1*sin(theta) + ax2*cos(2*theta) + bx2*sin(2*theta)
    B[1] = ay0 + ay1*cos(theta) + by1*sin(theta) + ay2*cos(2*theta) + by2*sin(2*theta)
    B[2] = az0 + az1*cos(theta) + bz1*sin(theta) + az2*cos(2*theta) + bz2*sin(2*theta)

    return B


  theta = pi / 2.0
  print MagFit(theta) - OldMagFitOrder2(theta)
  print MagFit(theta, order=3) - OldMagFitOrder3(theta)

  import os
  Ax,Ay,Az,triangles = getVertsTris(os.path.join('Asteroids','ida_m.obj'))
  print( (Ax,Ay,Az,triangles[-5:],) )
