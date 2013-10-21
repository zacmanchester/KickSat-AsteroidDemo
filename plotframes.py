import matplotlib
matplotlib.use('TkAgg') # do this before importing pylab
import matplotlib.pyplot as plt
import Image

fig = plt.figure()
ax = fig.add_subplot(111)

dtheta = 3
filenames = []
for k in range(0, 360, dtheta):
    filenames.append('./Frames/Frame'+str(k)+'.png')

def animate():
    im=plt.imshow(Image.open(filenames[0]))
    for filename in filenames[1:]:
        image=Image.open(filename)
        im.set_data(image)
        fig.canvas.draw() 

win = fig.canvas.manager.window
fig.canvas.manager.window.after(100, animate)
plt.show()