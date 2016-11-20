#!/usr/bin/env python

"""Implements a glipmse sensor and a glimpse network

Implements a glipmse sensor as described in 
https://arxiv.org/pdf/1406.6247v1.pdf
"""

import cv2
import numpy as np
import tensorflow as tf

__author__ = "Mohan Karthik"
__copyright__ = "Copyright 2016 - 2017, The RAM project for steering \
					prediction"
__credits__ = ["Mohan Karthik"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Mohan Karthik"
__email__ = "nemesis.nitt@gmail.com"
__status__ = "Prototype"

"""
Implements the Glimpse Sensor
"""
class GlimpseSensor:
	
	"""
	Consturctor
	
	src_img_sz: The size of the source image [w, h, d]
	
	core_img_sz: The size of the center image. All other images are
	set to repetetively being img_ratio times the size of the central 
	image [w', h', d]
	
	img_ratio: The ratio of the image resolutions compared to the 
	central image (float)
	
	num_imgs: Number of images the glimpse sensor must resolve the 
	incoming image into. The default configuration from the paper would
	be 3 (int)
	
	"""
	def __init__(self, src_img_sz, core_img_sz, img_ratio, num_imgs):
		
		# Preconditions
		assert len(src_img_sz) == 3
		assert len(core_img_sz) == 3
		
		# Check if the glimpse configuration is possible
		self.g_sz = np.ndarray((num_imgs, 3), dtype = np.uint32)
		for i in np.arange(num_imgs):
			self.g_sz[i][0] = core_img_sz[0] * (img_ratio ** (i))
			self.g_sz[i][1] = core_img_sz[1] * (img_ratio ** (i))
			self.g_sz[i][2] = core_img_sz[2]
			assert self.g_sz[i][0] <= src_img_sz[0]
			assert self.g_sz[i][1] <= src_img_sz[1]
		
		# Store the configurations
		self.core_sz = core_img_sz
		self.img_ratio = img_ratio
		self.num_imgs = num_imgs
		
	"""
	Implements the Glipmse sensor
	
	Takes an image input and a loc array [x, y] and returns a glipmse
	
	img: A color image of size src_img_sz (as specified in the 
	constructor
	
	loc: The central location around which to take the glipmse [x,y]
	
	returns: numpy array of (size num_imgs, core_img_sz)
	"""
	def Glimpse(self, img, loc):
		# Define the resulting glipmse image
		res = np.ndarray((self.num_imgs, self.core_sz[0], \
				self.core_sz[1], self.core_sz[2]), dtype=np.float32)
		
		# Loop through the number of images
		for i in np.arange(self.num_imgs):
			# Resize & splice the image into the result
			res[i,:,:] = np.atleast_3d(cv2.resize(img[
				(int)(loc[1] - (self.g_sz[i][1] / 2)):
				(int)(loc[1] + (self.g_sz[i][1] / 2)),
				(int)(loc[0] - (self.g_sz[i][0] / 2)):
				(int)(loc[0] + (self.g_sz[i][0] / 2)),:], 
				(self.core_sz[1], self.core_sz[0])))
	
		return res
		
"""
Implements the Glimpse network
"""
class GlimpseNetwork:
	"""
	Constructs a Glimpse network with the following parameters
	
	graph: An existing tensorflow graph
	bath_size: The batch size of each minibatch
	img_sz: The size of the glimpse image, after it is flattened
	hlg_dim: The dimension of the hl and hg network
	g_dim: The dimension of the g network
	"""
	
	def __init__(self, graph, batch_size, img_sz, hlg_dim, g_dim):
		# Continue using the existing graph
		with graph.as_default():
			# Add a placeholder for the input parameters
			with tf.name_scope("gnet_input") as scope:
				self.gnet_img = tf.placeholder(tf.float32, 
					shape=(batch_size, img_sz))
				self.gnet_loc = tf.placeholder(tf.float32,
					shape=(batch_size, 2))
			
			# Form the hg layer
			with tf.name_scope("gnet_hg") as scope:
				self.hg_w = tf.Variable(tf.truncated_normal(
					[img_sz, hlg_dim], 1 / np.sqrt(img_sz)), 
					name="gnet_hg_w")
				self.hg_b = tf.Variable(tf.zeros([hlg_dim]),
					name="gnet_hg_b")
				self.hg = tf.nn.relu(tf.nn.bias_add(tf.matmul(
					self.gnet_img, self.hg_w), self.hg_b))
				
			# Form the hl layer
			with tf.name_scope("gnet_hl") as scope:
				self.hl_w = tf.Variable(tf.truncated_normal(
					[2, hlg_dim], 1 / np.sqrt(2)), 
					name="gnet_hl_w")
				self.hl_b = tf.Variable(tf.zeros([hlg_dim]),
					name="gnet_hl_b")
				self.hl = tf.nn.relu(tf.nn.bias_add(tf.matmul(
					self.gnet_loc, self.hl_w), self.hl_b))
			
			# Form the g layer
			with tf.name_scope("gnet_g") as scope:
				self.g_in = tf.add(self.hg, self.hl)
				self.g_w = tf.Variable(tf.truncated_normal(
					[hlg_dim, g_dim], 1 / np.sqrt(hlg_dim)), 
					name="gnet_g_w")
				self.g_b = tf.Variable(tf.zeros([g_dim]),
					name="gnet_g_b")
				self.g = tf.nn.relu(tf.nn.bias_add(tf.matmul(
					self.g_in, self.g_w), self.g_b))
			
"""
Simple unit tests
"""
def test():
	# Get an image
	img = np.atleast_3d(cv2.imread(
		"/sharefolder/sdc-data/extract/center/1475186995013817919.png", 
		0))

	# Initialize the sensor
	mySensor = GlimpseSensor([480, 640, 1], [120, 160, 1], 2.0, 3)

	# Get a glimpse of the image
	glimpse = mySensor.Glimpse(img, [240, 320])

	# Create a tf graph
	graph = tf.Graph()

	# Form the glimpse network
	myNetwork = GlimpseNetwork(graph, 128, 57600, 128, 256)
	print ("Network Created")

	# Write the glipmses back
	for i in np.arange(3):
		cv2.imwrite('glimpse'+str(i)+'.png',glimpse[i,:,:,:])

# Script to execute the main
if __name__ == "__main__":
    test()
