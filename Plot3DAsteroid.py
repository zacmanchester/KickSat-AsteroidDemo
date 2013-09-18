from numpy import *
from mayavi.mlab import *
from serial import *

#Data lists for parsing
xdata = []
ydata = []
zdata = []
triangles = []

#Open serial port
#port1 = Serial("/dev/tty.uart-EEFF4676258B1340", 9600)

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
@animate(delay=100)
def anim():
	f = gcf()
	while 1:
		f.scene.camera.azimuth(getAngle())
		f.scene.render()
		yield

def getAngle():
	#line = port1.readline()
	#return int(line)
	return 10

anim()
