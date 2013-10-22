from math import *
from numpy import *
from scipy.optimize import minimize_scalar
from serial import *
from collections import deque
import pygame
import pygame.time
from pygame.locals import *
import threading

#Global Variables
dtheta = 3 #Angle (in degrees) between frames
Bias = array([0.0, 0.0, 0.0])
buf = deque([0.0]*10, 10)

#Initialize the display
pygame.init()
screen=pygame.display.set_mode((640,480),pygame.FULLSCREEN,16)
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
def AngleSolver(Bmeas, guess, lb, ub):

	def cost(theta):
		Bg = MagFit(theta)
		e = B - Bg
		return e.dot(e)

	theta = minimize_scalar(cost, bounds=(lb, ub), method='bounded')

	return theta

def readSerial():
	for line in Serial("/dev/cu.uart-EEFF4676258B1340", 9600):
		data = line.split()
		if len(data) == 3:
			try:
				B = array([float(data[0]), float(data[1]), float(data[1])])
				guess = buf[-1] + .2*(buf[-1]-buf[-6])
				lb = guess - pi/4
				ub = guess + pi/4
				
				res = AngleSolver(B, guess, lb, ub)

				if res < 0
					res = res + 2*pi
				if res > 2*pi
					res = res - 2*pi

				buf.append(res.x)
			except:
				print "Bad Data"

#Kick off the serial reader / angle solver thread
serialReaderThread = threading.Thread(target=readSerial)
serialReaderThread.start()

#Open Serial Port
port = Serial("/dev/cu.uart-0CFF4695F70D3A3F", 9600)

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

#Start the serial port reading/processing thread
serialReaderThread = threading.Thread(target=readSerial)
serialReaderThread.start()

#Display the first frame
screen.blit(frames[0],(0,0)) 
pygame.display.update()

#The animation loop
while true:
	index = round((60/pi)*buf.pop())
	screen.blit(frames[index])
	pygame.display.update()
	pygame.time.wait(80)
