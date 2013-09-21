from math import *
from numpy import *
from scipy.optimize import minimize_scalar
from mayavi.mlab import *
from serial import *
from collections import deque
import threading

#Global Variables
#last_theta = pi
buf = deque([0.0]*10, 10)

#Data lists for parsing
xdata = []
ydata = []
zdata = []
triangles = []

@show
@animate(delay=100)
def anim():
	f = gcf()
	while 1:
		view(azimuth=(180/pi)*buf.pop())
		f.scene.render()
		yield

def readSerial():
	for line in Serial("/dev/cu.uart-EEFF4676258B1340", 9600):
		data = line.split()
		if len(data) == 3:
			#try:
			B = array([float(data[0]), float(data[1]), float(data[1])])
			guess = buf[-1] + (buf[-1] - buf[-2])
			lb = guess - pi/18
			ub = guess + pi/18
			
			res = AngleSolver(B, guess, lb, ub)

			buf.append(res.x)
			#except:
			#	print "Bad Data"

#A 2-term Fourrier series fit to the B vector as a function of rotation angle 
def MagFit(theta):

	B = zeros(3)

	ax0 = -7.397
	ax1 = -8.329
	bx1 =  46.7
	ax2 = -2.102
	bx2 = -10.77

	ay0 =  26.04
	ay1 = -23.6
	by1 = -10.09
	ay2 =  7.748
	by2 = -0.7238

	az0 =  8.147
	az1 = -9.776
	bz1 =  8.226
	az2 = -0.2044
	bz2 = -2.14

	B[0] = ax0 + ax1*cos(theta) + bx1*sin(theta) + ax2*cos(2*theta) + bx2*sin(2*theta)
	B[1] = ay0 + ay1*cos(theta) + by1*sin(theta) + ay2*cos(2*theta) + by2*sin(2*theta)
	B[2] = az0 + az1*cos(theta) + bz1*sin(theta) + az2*cos(2*theta) + bz2*sin(2*theta)

	return B

#The derivative of the above function w.r.t. rotation angle
def DMagFit(theta):

	dB = zeros(3)

	ax1 = -8.329
	bx1 =  46.7
	ax2 = -2.102
	bx2 = -10.77

	ay1 = -23.6
	by1 = -10.09
	ay2 =  7.748
	by2 = -0.7238

	az1 = -9.776
	bz1 =  8.226
	az2 = -0.2044
	bz2 = -2.14

	dB[0] = -ax1*sin(theta) + bx1*cos(theta) - 2*ax2*sin(2*theta) + 2*bx2*cos(2*theta)	
	dB[1] = -ay1*sin(theta) + by1*cos(theta) - 2*ay2*sin(2*theta) + 2*by2*cos(2*theta)
	dB[2] = -az1*sin(theta) + bz1*cos(theta) - 2*az2*sin(2*theta) + 2*bz2*cos(2*theta)

	return dB

#Use Newton's method to solve for theta given B
def AngleSolver(B, guess, lb, ub):

	def cost(theta):
		Bg = MagFit(theta)
		e = B - Bg
		return e.dot(e)

	theta = minimize_scalar(cost, bounds=(lb, ub), method='bounded')

	return theta

#Parse OBJ file
with open("./Documents/GitHub/KickSat-AsteroidDemo/Asteroids/ida_m.obj") as f:
	for line in f:
		data = line.split()
		if len(data):
			if data[0] == 'v':
				xdata.append(float(data[1]))
				ydata.append(float(data[2]))
				zdata.append(float(data[3]))
			if data[0] == 'f':
				triangles.append((int(data[1].split("//")[0])-1, int(data[2].split("//")[0])-1, int(data[3].split("//")[0])-1))

#Convert virtex data to NumPy arrays
Ax = array(xdata)
Ay = array(ydata)
Az = array(zdata)

#Set up the figure with black background and white foreground
fig = figure("AsteroidFig", (0, 0, 0), (1, 1, 1))

#Draw the asteroid
triangular_mesh(Ax, Ay, Az, triangles, color=(.58, .58, .58))

#Start the serial port reading/processing thread
serialReaderThread = threading.Thread(target=readSerial)
serialReaderThread.start()

#Start the animation in the main thread
anim()
