import sys
import math
import AsteroidUtilities as au
import matplotlib.pyplot as plt

order = 3

modulus = 1e9

if __name__ == "__main__":
  for arg in sys.argv[1:]:

    if arg[:8] == "--order=":
      order = int(arg[8:])
      continue

    if arg == "--modulus2pi":
      modulus = 2.0 * math.pi
      continue

    print( "\nOpening %s" % (arg,) )
    coeffs,thetas,ys = au.FourierFit(arg,order=order,returnAll=True)
    print(coeffs)
    if thetas.size > 7:
      print(thetas[:3])
      print('...')
      print(thetas[-3:])
    else:
      print(thetas)
    print(ys)

    ms = au.MagFit(thetas,mtx=coeffs)
    lastC = 'bZ'
    for i,c in enumerate('rXy gYc bZm'.split()):
      plt.plot(thetas % modulus, ys[:,i],c[0]+'.',label=c[1]+' data')
      plt.plot(thetas % modulus, ms[:,i],c[2]    ,label=c[1]+' data')
      lastC = c
    plt.title(arg)
    plt.legend()
    plt.show()

