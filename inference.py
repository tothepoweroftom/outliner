import tensorflow as tf
import numpy as np
import os
import imageio

from scipy import misc

import argparse
import sys
from PIL import ImageOps, Image
import random

g_mean = np.array(([126.88,120.24,112.19])).reshape([1,1,3])
output_folder = "./test_output"

def rgba2rgb(img):
	return img[:,:,:3]*np.expand_dims(img[:,:,3],2)

def rgb2rgba(img):
	return np.dstack( ( img, 255*np.ones((img.shape[0], img.shape[1])) ) )



def simple_threshold(im, threshold=128):
    return ((im > threshold) * 255).astype("uint8")

def main(args):
	
	if not os.path.exists(output_folder):
		os.mkdir(output_folder)	
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = args.gpu_fraction)
	with tf.Session(config=tf.ConfigProto(gpu_options = gpu_options)) as sess:
		saver = tf.train.import_meta_graph('./meta_graph/my-model.meta')
		saver.restore(sess,tf.train.latest_checkpoint('./salience_model'))
		image_batch = tf.get_collection('image_batch')[0]
		pred_mattes = tf.get_collection('mask')[0]

		rgb = imageio.imread(args.rgb)
		num = random.randint(1,12)
		bg = Image.open("./backgrounds/shape-"+str(num)+".png")
		background = np.array(bg)
		rgb_ = rgb

		if rgb.shape[2]==4:
			rgb = rgba2rgb(rgb)
		
		if rgb_.shape[2]==3:
			rgba = rgb2rgba(rgb_)
			rgb_ = rgba

		
		origin_shape = rgb.shape[:2]
		img = np.zeros(rgb.shape,dtype=np.uint8)
		img.fill(255) # or img[:] = 255
		rgb = np.expand_dims(misc.imresize(rgb.astype(np.uint8),[320,320,3],interp="nearest").astype(np.float32)-g_mean,0)

		feed_dict = {image_batch:rgb}
		pred_alpha = sess.run(pred_mattes,feed_dict = feed_dict)
		final_alpha = misc.imresize(np.squeeze(pred_alpha),origin_shape)
		final_alpha = simple_threshold(final_alpha, threshold=150)

		final_alpha = final_alpha/255
		mask = final_alpha.reshape(*final_alpha.shape, 1)
		# resize background
		background = misc.imresize(background, origin_shape)
	


		# Here we add the images

		print(mask.shape)
		print(background.shape)
		print(rgb_.shape)


		blended = ((1.-mask)*background) + (mask*rgb_)
		imageio.imsave(os.path.join(output_folder,'alpha2.png'),blended)

def parse_arguments(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument('--rgb', type=str,
		help='input rgb',default = None)
	parser.add_argument('--rgb_folder', type=str,
		help='input rgb',default = None)
	parser.add_argument('--gpu_fraction', type=float,
		help='how much gpu is needed, usually 4G is enough',default = 1.0)
	return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
