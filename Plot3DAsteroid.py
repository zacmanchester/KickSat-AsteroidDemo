from math import *
from numpy import *
from mayavi.mlab import *
from serial import *

#Data lists for parsing
xdata = []
ydata = []
zdata = []
triangles = []

#Open serial port
port1 = Serial("/dev/cu.uart-EEFF4676258B1340", 9600)

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

@show
@animate(delay=50)
def anim():
	f = gcf()
	while 1:
		view(azimuth=getAngle())
		f.scene.render()
		yield

def getAngle():
	line = port1.readline()
	data = line.split()
	B = array([float(data[1]), float(data[2])])
	theta = (180/pi)*AngleSolver(B)
	print theta
	return theta

def MagXY(theta):

	#Scale 0<theta<2*pi to match the polynomial fit
	if theta > 2*pi:
		theta = fmod(theta, 2*pi)
	if theta < 0:
		theta = fmod(theta, 2*pi) + 2*pi

	px1 =    -0.04701;
	px2 =       1.069;
	px3 =      -9.452;
	px4 =       40.34;
	px5 =      -78.67;
	px6 =       27.86;
	px7 =        75.8;
	px8 =      -2.051;

	py1 =    0.007179;
	py2 =     -0.2688;
	py3 =        3.53;
	py4 =      -21.99;
	py5 =       69.15;
	py6 =      -95.76;
	py7 =       16.03;
	py8 =       54.82;

	Bx = px1*theta**7 + px2*theta**6 + px3*theta**5 + px4*theta**4 + px5*theta**3 + px6*theta**2 + px7*theta + px8;
	By = py1*theta**7 + py2*theta**6 + py3*theta**5 + py4*theta**4 + py5*theta**3 + py6*theta**2 + py7*theta + py8;

	dBx = 7*px1*theta**6 + 6*px2*theta**5 + 5*px3*theta**4 + 4*px4*theta**3 + 3*px5*theta**2 + 2*px6*theta + px7;
	dBy = 7*py1*theta**6 + 6*py2*theta**5 + 5*py3*theta**4 + 4*py4*theta**3 + 3*py5*theta**2 + 2*py6*theta + py7;

	return [array([Bx, By]), array([dBx, dBy])]

def AngleSolver(B):
	theta = pi

	for k in range(0,10):
		mag = MagXY(theta)
		Bg = mag[0]
		dBg = mag[1]

		e = B - Bg
		e2 = e.dot(e)
		de2dth = 2*e.dot(dBg)

		theta = theta + e2/de2dth

	return theta

anim()
