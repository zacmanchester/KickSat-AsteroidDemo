"""MagFitting.py

Test AsteroidUtilities.FourierFit() and plot results

Usage:

  python MagFitting.py [--order=N] [--noplot] [--plotphase] [RawData.txt[ Fits.txt[ ...]]]

Command-line arguments:

  --order=N     Highest order of Fourier coeffients (sin|cos(N*theta)); dflt=2

  --noplot      Do not plot results; default is to plot results

  --plotphase   Plot angles modulo 2PI on abscissa i.e. as phase values

  filepath(s)   flat ASCII file(s) with Mag data, one XYZ reading per line
                - last three columns are X, Y, Z data
                - first column, if four columns are present, is angle data
                  - if only three columns, first:last lines are angle=0:10PI
"""
if __name__ == "__main__":
  import os
  import sys
  import math
  import AsteroidUtilities as au

  order = 3
  modulus = 1e30
  noplot = False

  ### Loop over input arguments, print, and optionally plot, for each data file
  for arg in sys.argv[1:]:

    ### Parse --order=N argument
    if arg[:8] == "--order=":
      order = int(arg[8:])
      continue

    ### Parse --plotphase argument:  plot angles modulo 2*PI
    if arg == "--plotphase":
      modulus = 2.0 * math.pi
      continue

    ### Parse --noplot argument
    if arg == "--noplot":
      noplot = True
      continue

    ### To here, arg is filename of [theta,]X,Y,Z mag data, in columns
    print( "\nOpening %s" % (arg,) )

    ### - Convert data to coefficient matrix, theta values, y values
    coeffs,thetas,ys = au.FourierFit(arg,order=order,returnAll=True)

    ### - Show coefficients
    print(coeffs)

    if "MAGFITDEBUG" in os.environ:

      ### - Show data (optional)
      if thetas.size > 7:
        print(thetas[:3])
        print('...')
        print(thetas[-3:])
      else:
        print(thetas)

      print(ys)


    if noplot: continue

    ### Do plots:

    import matplotlib.pyplot as plt

    ### - Calculate model values, which should approximate y values
    ###   (variable ys) from coefficients
    ms = au.MagFit(thetas,mtx=coeffs)

    ### - For each axis, plot input data as points, model data as lines
    ###   - Xry:  X data; red nput data points; yellow model data lines
    ###   - Ygc:  Y data; green nput data points; cyan model data lines
    ###   - Zbm:  Z data; blue nput data points; magenta model data lines
    for i,c in enumerate('Xry Ygc Zbm'.split()):
      plt.plot(thetas % modulus, ys[:,i],c[1]+'.',label=c[0]+' data')
      plt.plot(thetas % modulus, ms[:,i],c[2]    ,label=c[0]+' model')

    plt.xlabel('%s, radians' % ('Angle' if modulus is 1e30 else 'Phase'))
    plt.ylabel('Magnetometer, arbitrary units')
    plt.title(arg)
    plt.legend()
    plt.show()

