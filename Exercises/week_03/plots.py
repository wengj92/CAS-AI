import numpy as np
import matplotlib.pyplot as plt
import torchvision


def show_images(images):
    img = torchvision.utils.make_grid(images)
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# EOF