from math import *
from numpy import *
from mayavi.mlab import *
from serial import *

#Image Size
size_x = 400
size_y = 300

#Data lists for parsing
xdata = []
ydata = []
zdata = []
triangles = []

#Parse OBJ file
with open("./Asteroids/ida_m.obj") as f:
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
fig = figure("AsteroidFig", (0, 0, 0), (1, 1, 1), size=(size_x, size_y+28))

#Draw the asteroid
triangular_mesh(Ax, Ay, Az, triangles, color=(.58, .58, .58))

@show
@animate(delay=50)
def anim():
	f = gcf()
	theta = 0
	while theta < 360:
		view(azimuth=theta)
		savefig("./Frames_"+str(size_x)+"/Frame"+str(theta)+".png", figure=fig, magnification=1)
		theta = theta+3
		f.scene.render()

		yield

anim()
