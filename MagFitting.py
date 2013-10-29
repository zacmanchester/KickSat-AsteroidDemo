import sys
import AsteroidUtilities as au
import matplotlib.pyplot as plt

order = 3

if __name__ == "__main__":
  for arg in sys.argv[1:]:
    if arg[:8] == "--order=":
      order = int(arg[8:])
      continue
    print( "\nOpening %s" % (arg,) )
    print( au.FourierFit(arg,order=order) )
    
