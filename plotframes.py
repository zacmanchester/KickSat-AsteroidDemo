import matplotlib
matplotlib.use('TkAgg') # do this before importing pylab
import matplotlib.pyplot as plt
import Image
import glob

fig = plt.figure()
ax = fig.add_subplot(111)

def animate():
    filenames=sorted(glob.glob('./Frames/*.png'))
    im=plt.imshow(Image.open(filenames[0]))
    for filename in filenames[1:]:
        image=Image.open(filename)
        im.set_data(image)
        fig.canvas.draw() 

win = fig.canvas.manager.window
fig.canvas.manager.window.after(100, animate)
plt.show()