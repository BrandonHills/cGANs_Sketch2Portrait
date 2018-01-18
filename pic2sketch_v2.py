import argparse
import os
import cv2
import numpy as np


def dodgeV2(image, mask):
  	return cv2.divide(image, 255-mask, scale=256)

# def burnV2(image, mask):
# 	ans = 255 – cv2.divide(255-image, 255-mask, scale=256)
#   	return ans

def sketchify(img_rgb):

	maxIntensity, blur, phi, theta = 255.0, 0, 3, 1
	
	img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

	image = img_gray
	darker_img = (maxIntensity/phi)*(image/(maxIntensity/theta))**2
	darker_img = np.array(darker_img,dtype="uint8")

	img_gray_inv = 255 - darker_img

	img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(21, 21),
	                            sigmaX=blur, sigmaY=blur)

	img_blend = dodgeV2(darker_img, img_blur)

	return img_blend



def main():

	parser = argparse.ArgumentParser('create image pairs')
	parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='faces/')
	# parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='faces/B')
	parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default='faces_sketchv2/')
	parser.add_argument('--num_imgs', dest='num_imgs', help='number of images',type=int, default=1000000)
	parser.add_argument('--use_AB', dest='use_AB', help='if true: (0001_A, 0001_B) to (0001_AB)',action='store_true')
	args = parser.parse_args()


	# Telling the user which args were chosen.
	print("DISPLAYING ARGS:")
	for arg in vars(args):
	    print('[%s] = ' % arg,  getattr(args, arg))

	images = os.listdir(args.fold_A)
	print("images:", images)
	
	if ".DS_Store" in images:
		print(".DS_Store Removed...")
		images.remove(".DS_Store")

	for img_path in images:
		origin_path = os.path.join(args.fold_A, img_path)
		print("origin_path:", origin_path)
		img_rgb = cv2.imread(origin_path)
		sketch = sketchify(img_rgb)

		path = os.path.join(args.fold_A, img_path)
		path_Sketch = os.path.join(args.fold_AB, img_path)

		if os.path.isfile(path) and not os.path.isfile(path_Sketch):
			
			print("path_Sketch:", path_Sketch)

			cv2.imwrite(path_Sketch, sketch)

main()



