# cannot easily visualize filters lower down
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
import numpy as np

# load the model
model = VGG16()

# summarize the model
model.summary()

pause = input("Presione Enter para continuar...")

# summarize filter shapes
for layer in model.layers:
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# get filter weights
	filters, biases = layer.get_weights()
	print(layer.name, filters.shape)

# retrieve weights from the second hidden layer
filters, biases = model.layers[1].get_weights()
# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

print("Visualizar los filtros:")
visual = input("Presione 'Y' para visualizar como imagen. Presione 'N' para visualizar como texto")

n_filters, ix = 6, 1

if(visual == 'Y'):
	# plot first 6 filters
	for i in range(n_filters):
		# get the filter
		f = filters[:, :, :, i]
		# plot each channel separately
		for j in range(3):
			# specify subplot and turn of axis
			ax = pyplot.subplot(n_filters, 3, ix)
			ax.set_xticks([])
			ax.set_yticks([])
			# plot filter channel in grayscale
			pyplot.imshow(f[:, :, j], cmap='gray')
			ix += 1
	# show the figure
	pyplot.savefig(str('filters.jpg'), bbox_inches='tight')
	#pyplot.show()
	pyplot.clf()

if(visual == 'N'):
	layer = model.layers[1]
	print(layer.name, filters.shape)
	# print first 6 filters
	for i in range(n_filters):
		# get the filter
		f = filters[:, :, :, i]
		print('Filtro: ' ,i+1)
		for j in range(len(f)):
			print('Channel: ' ,j+1)
			print(f[:,])

pause = input("Presione Enter para continuar...")
	
# Visualizar Feature Maps
model_p = Model(inputs=model.inputs, outputs=model.layers[1].output)
model_p.summary()
# load the image with the required shape
img = load_img('bird.jpg', target_size=(224, 224))
# convert the image to an array
img = img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
img = preprocess_input(img)
# get feature map for first hidden layer
feature_maps = model_p.predict(img)

print("Visualizar los mapas:")
visual = input("Presione 'Y' para visualizar como imagen. Presione 'N' para visualizar como texto")

if(visual == 'Y'):
	# plot all 64 maps in an 8x8 squares
	square = 8
	ix = 1
	for _ in range(square):
		for _ in range(square):
			# specify subplot and turn of axis
			ax = pyplot.subplot(square, square, ix)
			ax.set_xticks([])
			ax.set_yticks([])
			# plot filter channel in grayscale
			pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
			ix += 1
	# show the figure
	pyplot.savefig(str('feature_maps.jpg'), bbox_inches='tight')
	#pyplot.show()
	pyplot.clf()

if(visual == 'N'):
	# print feature map values from first filter and first channel
	#print(np.shape(feature_maps))
	f_m = feature_maps[0, :, :, 0]
	print('Tamaño del mapa: ', np.shape(f_m))
	print(f_m)
	
	

