from math import *
from numpy import *
from scipy.optimize import minimize_scalar
from serial import *
from collections import deque
import pygame
from pygame.locals import *
import threading

#Global Variables
#last_theta = pi
buf = deque([pi]*10, 10)

#Data lists for parsing
xdata = []
ydata = []
zdata = []
triangles = []



def readSerial():
	for line in Serial("/dev/cu.uart-EEFF4676258B1340", 9600):
		data = line.split()
		if len(data) == 3:
			#try:
			B = array([float(data[0]), float(data[1]), float(data[1])])
			guess = buf[-1] + .2*(buf[-1]-buf[-6])
			lb = guess - pi/4
			ub = guess + pi/4
			
			res = AngleSolver(B, guess, lb, ub)

			buf.append(res.x)
			#except:
			#	print "Bad Data"

#A 2-term Fourrier series fit to the B vector as a function of rotation angle 
def MagFit(theta):

	B = zeros(3)

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

	B[0] = ax0 + ax1*cos(theta) + bx1*sin(theta) + ax2*cos(2*theta) + bx2*sin(2*theta) + ax3*cos(3*theta) + bx3*sin(3*theta)
	B[1] = ay0 + ay1*cos(theta) + by1*sin(theta) + ay2*cos(2*theta) + by2*sin(2*theta) + ay3*cos(3*theta) + by3*sin(3*theta)
	B[2] = az0 + az1*cos(theta) + bz1*sin(theta) + az2*cos(2*theta) + bz2*sin(2*theta) + az3*cos(3*theta) + bz3*sin(3*theta)

	return B

#Use Newton's method to solve for theta given B
def AngleSolver(B, guess, lb, ub):

	def cost(theta):
		Bg = MagFit(theta)
		e = B - Bg
		return e.dot(e)

	theta = minimize_scalar(cost, bounds=(lb, ub), method='bounded')

	return theta

#Start the serial port reading/processing thread
serialReaderThread = threading.Thread(target=readSerial)
serialReaderThread.start()

#Start the animation in the main thread

