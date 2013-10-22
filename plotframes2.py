import matplotlib

matplotlib.use('TkAgg') # do this before importing pylab
matplotlib.rcParams['toolbar'] = 'None'

import matplotlib.pyplot as plt
import scipy.misc as misc

fig = plt.figure()
ax = fig.add_subplot(111)
plt.axis('off')

dtheta = 3
frames = []
for k in range(0, 360, dtheta):
    filename = './Frames/Frame'+str(k)+'.png'
    frames.append(misc.imread(filename))

def animate():
    im=plt.imshow(frames[0])
    for frame in frames[1:]:
        im.set_data(frame)
        fig.canvas.draw() 

win = fig.canvas.manager.window
fig.canvas.manager.window.after(100, animate)
plt.show()
