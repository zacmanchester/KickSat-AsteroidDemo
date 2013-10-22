from math import *
from numpy import *
from scipy.optimize import fminbound
from serial import *
from collections import deque
import pygame
import pygame.time
from pygame.locals import *
import threading

#Global Variables
dtheta = 3 #Angle (in degrees) between frames
Bias = array([0.0, 0.0, 0.0])
lastAngle = 0

#Initialize the display
pygame.init()
screen=pygame.display.set_mode((640,480),pygame.FULLSCREEN,16)
#screen=pygame.display.set_mode((640,480),0,16)
pygame.display.set_caption("KickSat Asteroid Tracker")

#Read all the frames from their PNG files
frames = []
for k in range(0, 360, dtheta):
    filename = './Frames_640/Frame'+str(k)+'.png'
    frames.append(pygame.image.load(filename).convert())

#A 2-term Fourrier series fit to the B vector as a function of rotation angle
#Zero is defined as the asteroid pointing at the viewer
def MagFit(theta):

	B = zeros(3)

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

#Solve for theta given B
def AngleSolver(B, guess, lb, ub):

	def cost(theta):
		Bg = MagFit(theta)
		e = B - Bg
		return e.dot(e)

	theta = fminbound(cost, lb, ub)

	return theta

def getAngle():
	while True:
		line = port.readline()
		#print(line)
		data = line.split()
		if len(data) == 3:
			try:
				B = array([float(data[0]), float(data[1]), float(data[1])])
				lb = lastAngle - pi/6
				ub = lastAngle + pi/6
			
				res = AngleSolver(B, lastAngle, lb, ub)
				if res < 0:
					res = res + 2*pi
				if res > 2*pi:
					res = res - 2*pi
			
				#print(str(res))
				return res
			except:
				print "Bad Data"

#Open Serial Port
port = Serial("/dev/ttyO1", 9600)

#Wait to make sure we get some good data in the buffer
port.readline()
pygame.time.wait(2000)
port.flush()

#Grab a good measurement and assume we're at 0 degrees to calculate bias vector 
line = port.readline()
data = line.split()
while len(data) != 3:
	line = port.readline()
	data = line.split()
Bmeas = array([float(data[0]), float(data[1]), float(data[1])])
Bpred = MagFit(0.0)
Bias = Bpred - Bmeas

#Display the first frame
screen.blit(frames[0],(0,0)) 
pygame.display.update()

#The animation loop
while True:
	angle = getAngle()
	index = int(round((60/pi)*angle))%120
	screen.blit(frames[index],(0,0))
	pygame.display.update()
	lastAngle = angle
