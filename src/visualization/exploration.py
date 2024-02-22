import  matplotlib.pyplot as plt
import numpy as np
def imshow(img):
    print(type(img))
    img = img / 2 + 0.5 #if denormalization is needed
    npimg = np.array(img)
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()