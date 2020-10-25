from PIL import Image
import numpy as np

image=Image.open('C:/Users/Mohit K/Desktop/datasets/clown.png')

data=np.asarray(image)

kernelv=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])   #vertical
kernelh=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])  #horizontal
laplacian_filter = np.array([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]])
"""
def standardize(image):
    eps = 1e-5  # some fields have all 255 so variance will be 0, to avoid division by zero, introduced eps
    return (image - np.mean(image))/(np.std(image)+eps)
"""
#kernelvv = kernelv.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_LEFT_RIGHT)

#convolve with kernelv
a = np.zeros([254, 254], dtype = int)
for i in range(0,data.shape[0]-2):
    for j in range(0,data.shape[1]-2):

        a[i, j] = (kernelv * data[i: i+3, j: j+3]).sum()

im = Image.fromarray(a)
im.show()

#kernelhh = kernelh.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_LEFT_RIGHT)

#convolve with kernelh
b = np.zeros([254, 254], dtype = int)
for i in range(0,data.shape[0]-2):
    for j in range(0,data.shape[1]-2):

        b[i, j]= (kernelh * data[i: i+3, j: j+3]).sum()

im1 = Image.fromarray(b)
im1.show()

#magnitude

final=np.sqrt(np.square(a) + np.square(b))
final *= 255.0 / final.max()
print(final)

#index of max
ind = np.unravel_index(np.argmax(final, axis=None), final.shape)
print(ind)


im2=Image.fromarray(final)
im2.show()

########laplacian

c = np.zeros([254, 254], dtype = int)
for i in range(0,data.shape[0]-2):
    for j in range(0,data.shape[1]-2):

        c[i, j]= (laplacian_filter * data[i: i+3, j: j+3]).sum()

im3=Image.fromarray(c)
im3.show()

