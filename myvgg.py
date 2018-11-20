from keras.applications.vgg16 import VGG16
#from kerasvis.vis.utils import utils
from keras import activations
from keras.utils import plot_model

# Build the VGG16 network with ImageNet weights
model = VGG16(weights='imagenet',
                  include_top=True,
                  input_shape=(224, 224, 3))

plot_model(model, to_file='/home/bibhu/Desktop/squeezenet/vgg16/model.png')
