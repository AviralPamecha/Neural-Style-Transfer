import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template, send_file
from PIL import Image

from werkzeug.utils import secure_filename

import cv2
import io
import base64

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
np.random.seed(42)


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file1' and 'file2' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file1 = request.files['file1']
	file2 = request.files['file2']

	if file1.filename and file2.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file1 and allowed_file(file1.filename):
		filename1 = secure_filename(file1.filename)
		file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
	if file2 and allowed_file(file2.filename):
		filename2 = secure_filename(file2.filename)
		file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))	

		#print('upload_image filename: ' + filename)
		flash('Image successfully uploaded and displayed')
		content = plt.imread(app.config['UPLOAD_FOLDER']+filename1)
		style = plt.imread(app.config['UPLOAD_FOLDER']+filename2)

		def load_image(image):
			 image = plt.imread(image)
			 img = tf.image.convert_image_dtype(image, tf.float32)
			 img = tf.image.resize(img,[500, 500])
			 img = img[tf.newaxis, :]
			 return img
		content = load_image(app.config['UPLOAD_FOLDER']+filename1)
		style = load_image(app.config['UPLOAD_FOLDER']+filename2)
		vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
		vgg.trainable = False
		content_layers = ['block1_conv1']
		style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
		num_content_layers = len(content_layers)
		num_style_layers = len(style_layers)

		def mini_model(layer_names, model):
			outputs = [model.get_layer(name).output for name in layer_names]
			model = Model([vgg.input], outputs)
			return model
		def gram_matrix(tensor):
			temp = tensor
			temp = tf.squeeze(temp)
			fun = tf.reshape(temp, [temp.shape[2], temp.shape[0]*temp.shape[1]])
			result = tf.matmul(temp, temp, transpose_b=True)
			gram = tf.expand_dims(result, axis=0)
			return gram

		class Custom_Style_Model(tf.keras.models.Model):
			def __init__(self, style_layers, content_layers):
				super(Custom_Style_Model, self).__init__()
				self.vgg =  mini_model(style_layers + content_layers, vgg)
				self.style_layers = style_layers
				self.content_layers = content_layers
				self.num_style_layers = len(style_layers)
				self.vgg.trainable = False 

			def call(self, inputs):	
				inputs = inputs*255.0
				# Preprocess them with respect to VGG19 stats
				preprocessed_input = preprocess_input(inputs)
				# Pass through the mini network
				outputs = self.vgg(preprocessed_input)
				# Segregate the style and content representations
				style_outputs, content_outputs = (outputs[:self.num_style_layers],outputs[self.num_style_layers:])
				# Calculate the gram matrix for each laye
				style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
				# Assign the content representation and gram matrix in
				# a layer by layer fashion in dicts
				content_dict = {content_name:value for content_name, value in zip(self.content_layers, content_outputs)}
				style_dict = {style_name:value for style_name, value in zip(self.style_layers, style_outputs)}
				return {'content':content_dict, 'style':style_dict}


		# Note that the conte
		# content and style variables respectively
		extractor = Custom_Style_Model(style_layers, content_layers)
		style_targets = extractor(style)['style']
		content_targets = extractor(content)['content']
		opt = tf.optimizers.Adam(learning_rate=0.02)
		# Custom weights for style and content update
		style_weight=100
		content_weight=1
		# Custom weights for different style layers
		style_weights = {'block1_conv1': 1.,'block2_conv1': 0.8,'block3_conv1': 0.5,'block4_conv1': 0.3,'block5_conv1': 0.1}
		# We can Play around with weights

		def total_loss(outputs):
			style_outputs = outputs['style']
			content_outputs = outputs['content']
			style_loss = tf.add_n([style_weights[name]*tf.reduce_mean((style_outputs[name]-style_targets[name])**2) for name in style_outputs.keys()])
			# Normalize
			style_loss *= style_weight / num_style_layers
			content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) for name in content_outputs.keys()])
			# Normalize
			content_loss *= content_weight / num_content_layers
			loss = style_loss + content_loss
			return loss

		@tf.function()
		def train_step(image):
			with tf.GradientTape() as tape:
				# Extract the features
				outputs = extractor(image)
				# Calculate the loss
				loss = total_loss(outputs)
				# Determine the gradients of the loss function w.r.t the image pixels
				grad = tape.gradient(loss, image)
				# Update the pixels
				opt.apply_gradients([(grad, image)])
				# Clip the pixel values that fall outside the range of [0,1]
				image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

		target_image = tf.Variable(content)
		epochs = 1
		steps_per_epoch = 1
		step = 0
		for n in range(epochs):
			for m in range(steps_per_epoch):
				step += 1
				train_step(target_image)
				a = np.squeeze(target_image.read_value(), 0)
				img = cv2.convertScaleAbs(a, alpha=(255.0))
				img = Image.fromarray(img.astype('uint8'))
				file_object = io.BytesIO()
				img.save(file_object, 'PNG')
				file_object.seek(0)
				pngImageB64String = "data:image/png;base64,"
				pngImageB64String += base64.b64encode(file_object.getvalue()).decode('utf8')

					
		return render_template("result.html", image=pngImageB64String)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True)

















