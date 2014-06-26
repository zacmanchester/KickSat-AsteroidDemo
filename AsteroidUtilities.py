import math
import numpy as np
import scipy.optimize as so
import numpy.matrixlib as nm


########################################################################
### Module-local tuple of matrices of Fourier fit coefficents
### - Index within tuple is order of matrix
### - Each principal axis' (PA's) coefficients are on one line in this file
### - Note .T (transpose) at end of each so PA's coefs will be a column

mtxlist = ( None, None

### 2-term Fourier fit:       a0        a1     b1      a2      b2
          , nm.matrix( [ [-15.94,   -7.522, 48.97, -5.918, -3.235 ]  ### B[0]
                       , [  2.484, -22.72,  -9.989, 2.358, -4.198 ]  ### B[1]
                       , [ 34.93,   -2.168, 10.02, -1.838, -0.6224]  ### B[2]
                       ] ).T

### 3-term Fourier fit:        a0      a1       b1       a2    b2       a3       b3
          , nm.matrix( [ [  19.04, -37.98,   -0.4392, -0.4125, -8.412,  1.689,  0.5891]  ### B[0]
                       , [  13.87,   4.463, -19.73,    4.672,   1.369, -0.7784, 0.889 ]  ### B[1]
                       , [  -4.907, -8.234,   0.601,  -0.0403, -1.91,   0.442,  0.1858]  ### B[2]
                       ] ).T

          , )


########################################################################
def buildvec(theta, order):
  """
Build row vector of Fourier cos(n*Theta) and sin(n*Theta) terms to be
multiplied by coefficents in mtxlist tuple above;

Argument theta may be a list, tuple or np.ndarray
"""
  ### Force theta into np.ndarray where .shape = (N,1,)
  lclTheta = np.array(theta)
  shp = lclTheta.shape
  if shp:
    lclTheta = lclTheta.reshape((-1,1,))
  else:
    lclTheta = np.array([theta]).reshape((-1,1,))

  ### Order 1:
  ###   [cos(0*theta), cos(1*theta), sin(1*theta)]
  ###   - exclude sin(0*theta) (=0)
  vec = np.hstack([np.ones(lclTheta.size).reshape((-1,1,)), np.cos(lclTheta), np.sin(lclTheta)])

  ### Append one cosine and sine term pair per remaining order
  lclOrder = order - 1
  while lclOrder > 0:
    ###                   cos(nT+T) =                            ,   sin(nT+T) =
    ###                   cos(  nT)   *cos(  T)   - sin(  nT) *sin(  T),   cos(  nT)*sin(     T)   + sin(  nT)   *cos(  T)
    vec = np.hstack([vec, vec[:,-2:-1]*vec[:,1:2] - vec[:,-1:]*vec[:,2:3],   vec[:,-1:]*vec[:,1:2] + vec[:,-2:-1]*vec[:,2:3]])
    lclOrder -= 1
  return vec


########################################################################
def MagFit(theta,order=2,mtx=None):
  """
Build row vector of Fourier terms per chosen order from theta,
return row vector of dot products of that with each column of
matrix of chosen order.

Argument theta may be a list, tuple or np.ndarray
"""
  if mtx is None:
    mtx = mtxlist[order]
  else:
    order = mtx.shape[0] >> 1

  rtn = buildvec(theta,order) * mtx

  if np.array(theta).shape: return rtn
  return rtn.getA1()


########################################################################
def AngleSolver(B, guess, lb, ub, order=2, useFmin=True, tol=1e-9):
  """
Solve for theta to fit row vector B input to MagFit(theta)

Uses scipy.optimize.fminbound(...) if useFmin is True; else use
.minimize_scalar(...,method='bounded')
*** N.B. the former, fminbound, is a wrapper for the latter

Return:

  theta that minimizes |B - MagFit(theta,order)| (magnitude)

Arguments:

  B        row vector to fit with theta via MagFit(theta)
  guess    ignored
  lb,ub    lower and upper bounds, respectively
  order    which mtxlist[order] matrix to use
  useFmin  True to use fminbound
  tol      tolerance
"""

  ######################################################################
  def cost(theta):
    """Cost function to be minimized by AngleSolver via varying theta;
       returns distance betwween input 3-vectors B and MagFit(theta)."""
    e = B - MagFit(theta,order=order)
    return e.dot(e)

  ### Use fminbound if useFmin is True (default) ...
  if useFmin: return so.fminbound(cost, lb, ub, xtol=tol)

  ### ... else use functionally identical minimize_scalar(...,method='bounded')

  return so.minimize_scalar(cost, bounds=(lb, ub,), method='bounded', tol=tol).x


########################################################################
def getVertsTris(filepath):
  """
Read shape from file in ASCII OBJ format
- parses v and f lines only
- ignores normal and texture vertices

Returns (Ax,Ay,Az,triangles,) tuple
"""
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


########################################################################
def FourierFit(filename,order=2,returnMatrixOnly=True,returnAll=False):
  """Open file, read columns:  theta (optional); Bx, By, Bz

     Calculate Fourier coefficients via least-squares; return matrix of
     coefficients or full np.linalg.lstsq() return value
"""
  rawdata = np.array([ [float(v) for v in line.strip().split()] for line in open(filename,"rb") if line.strip()])
  shp = rawdata.shape

  if shp[1] == 4:
    ### Extract thetas from first column
    thetas = rawdata[:,0:1]

  elif shp[1] == 3:
    ### assume first and last lines are theta = zero and ten PI
    thetas = math.pi * 10.0 * np.arange(shp[0], dtype=np.float).reshape((-1,1,)) / (shp[0]-1)

  ### Build cos(nT), sin(nT) terms, .shape = (N,2*order+1)
  thetaVecs = buildvec(thetas,order=order)

  ### Extract mag readings, .shape = (N,3)
  ys = rawdata[:,shp[1]-3:]

  ### Fit the data using least squares
  result = np.linalg.lstsq(thetaVecs, ys)

  
  ### Return matrix, or full .lstsq result
  rtn = (np.matrixlib.matrix(result[0]) if returnMatrixOnly else result,)
  return rtn + ((thetas,ys,) if returnAll else ())


########################################################################
### Test code
if __name__=="__main__":
  ### testing
  def OldMagFitOrder3(theta):
    """Copied from github.com/zacinaction ca. late Oct, 2013
       N.B. this can only take a scalar theta, unlike MagFit above"""

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

    B[0] = ax0 + ax1*math.cos(theta) + bx1*math.sin(theta) + ax2*math.cos(2*theta) + bx2*math.sin(2*theta) + ax3*math.cos(3*theta) + bx3*math.sin(3*theta)
    B[1] = ay0 + ay1*math.cos(theta) + by1*math.sin(theta) + ay2*math.cos(2*theta) + by2*math.sin(2*theta) + ay3*math.cos(3*theta) + by3*math.sin(3*theta)
    B[2] = az0 + az1*math.cos(theta) + bz1*math.sin(theta) + az2*math.cos(2*theta) + bz2*math.sin(2*theta) + az3*math.cos(3*theta) + bz3*math.sin(3*theta)

    return B


  def OldMagFitOrder2(theta):
    """Copied from github.com/zacinaction ca. late Oct, 2013
       N.B. this can only take a scalar theta, unlike MagFit above"""

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

    B[0] = ax0 + ax1*math.cos(theta) + bx1*math.sin(theta) + ax2*math.cos(2*theta) + bx2*math.sin(2*theta)
    B[1] = ay0 + ay1*math.cos(theta) + by1*math.sin(theta) + ay2*math.cos(2*theta) + by2*math.sin(2*theta)
    B[2] = az0 + az1*math.cos(theta) + bz1*math.sin(theta) + az2*math.cos(2*theta) + bz2*math.sin(2*theta)

    return B


  ### Compare new MagFit routine against 2013-Oct MagFit routines
  err2 = []
  err3 = []
  for i in range(3000):
    theta = i * math.pi / 600
    if i==0: print( MagFit(theta,order=2) , OldMagFitOrder2(theta) )
    err2 += [ [ [np.abs(v) for v in MagFit(theta,order=2) - OldMagFitOrder2(theta)], theta, 2] ]
    err3 += [ [ [np.abs(v) for v in MagFit(theta,order=3) - OldMagFitOrder3(theta)], theta, 3] ]

  print( "Testing Magfits ..." )
  errCount = 0
  for errs in [err2,err3]:
    errs.sort()
    for err in errs:
      if max(err[0]) <= 1e-13: continue
      print( dict(zip('errs theta order'.split(),err)) )
      errCount += 1

  if errCount==0: print( "  No errors > 1E-13 between old and new MagFits" )

  ### Repeat for array of thetas
  thetas = [i * math.pi for i in range(7)]
  err2 = MagFit(thetas,2) - np.vstack( [OldMagFitOrder2(theta) for theta in thetas] )
  err3 = MagFit(thetas,3) - np.vstack( [OldMagFitOrder3(theta) for theta in thetas] )
  if np.max(np.abs(err2)) > 1e-13: print( 'Errors > 1e-13 (order=2):', err2 )
  if np.max(np.abs(err3)) > 1e-13: print( 'Errors > 1e-13 (order=3):', err3 )

  ### Compare input and output thetas
  print( "Testing AngleSolver ..." )
  errCount = 0
  for order in (2,3):
    for i in range(3000):
      theta = i * math.pi / 500
      B = MagFit(theta,order)
      thetaSolve = AngleSolver(B, None, theta-math.pi/4, theta+math.pi/4, order=order, useFmin=(order==2))
      if abs(thetaSolve-theta) <= 1e-6: continue
      print( (order,i,theta,thetaSolve,thetaSolve-theta,) )
      errCount += 1

  if errCount==0: print( "  No errors > 1E-6 from AngleSolver" )

  ### Get Ida shape from file, print out some pieces of it
  import os
  print( "Testing OBJ reader ..." )
  Ax,Ay,Az,triangles = getVertsTris(os.path.join('Asteroids','ida_m.obj'))
  print( (Ax,Ay,Az,triangles[-5:],) )

