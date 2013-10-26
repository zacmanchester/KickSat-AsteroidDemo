import pygame
from pygame.locals import *
from sys import exit
 
pygame.init()
screen=pygame.display.set_mode((640,480),pygame.FULLSCREEN,16)
pygame.display.set_caption("KickSat Asteroid Tracker")

dtheta = 3
frames = []
for k in range(0, 360, dtheta):
    filename = './Frames_640/Frame'+str(k)+'.png'
    frames.append(pygame.image.load(filename).convert())

for frame in frames:
    screen.blit(frame,(0,0))
    pygame.display.update()
    
exit()

