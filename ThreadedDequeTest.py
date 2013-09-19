from math import *
from numpy import *
from serial import *
from collections import deque
import multiprocessing

#Global Variables
#last_theta = pi
buf = deque([0.0]*10, 10)

def readSerial():
	last_theta = pi

	for line in Serial("/dev/cu.uart-EEFF4676258B1340", 9600):
		data = line.split()
		print data
		if len(data) == 3:
			try:
				B = array([float(data[0]), float(data[1]), float(data[1])])
				theta = AngleSolver(B, pi)
				print theta
				buf.append(theta)
			except:
				print "Bad Data"

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
def AngleSolver(B, guess):
	theta = guess

	Bg = MagFit(theta)

	for k in range(0,5):

		dBg = DMagFit(theta)

		e = B - Bg
		J = sum(e*e)
		dJdth = 2*sum(e*dBg)

		alpha = 1
		for i in range(0,10):
			print i
			thetanew = theta + alpha*J/dJdth
			Bg = MagFit(thetanew)
			enew = B - Bg
			Jnew = sum(enew*enew)
			if Jnew < J:
				theta = thetanew
				break
			alpha = alpha/2

	return theta

#Start the serial port reading/processing thread
serialReaderProcess = multiprocessing.Process(target=readSerial)
serialReaderProcess.start()

print "Process Started."
