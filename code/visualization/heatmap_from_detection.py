import cv2, os, time, argparse
import numpy as np

from keras import backend as K

from yad2k.keras_yolo import yolo_eval, yolo_head, loadYoloModel

from core.BoundBox import BoundBox
from core.Utils import convertDetectionsToMyImp, loadAndNormalizeImg, image_resize
from core.DisplayUtils import put_title


def heat_map(activation_output, img, yolo_model):
	last_conv_layer = yolo_model.get_layer('leaky_re_lu_22')

	grads = K.gradients(activation_output, last_conv_layer.output)[0]
	pooled_grads = K.mean(grads, axis=(0, 1, 2))

	iterate = K.function([yolo_model.input], [pooled_grads, last_conv_layer.output[0]])

	pooled_grads_value, conv_layer_output_value = iterate([img])

	for j in range(1024):
		conv_layer_output_value[:, :, j] *= pooled_grads_value[j]

	heatmap = np.mean(conv_layer_output_value, axis=-1)
	heatmap = np.maximum(heatmap, 0)

	heatmap = cv2.resize(heatmap, (img[0].shape[1], img[0].shape[0]))
	return heatmap

def heatmap_title(title, activation_output, img, yolo_model):
	print(activation_output)
	ht = heat_map(yolo_model.output[activation_output], img, yolo_model)
	return put_title(title, ht)

def normalize_grads(grds, groups):
	print(grds)
	print(groups)
	
	biggestValue = 0 
	for group in groups:
		print("grupin", group)
		if np.max(grds[group]) > biggestValue:
			biggestValue = np.max(grds[group])
		   
	for img_name in groups:
		grds[img_name] /= biggestValue
		grds[img_name] = cv2.resize(grds[img_name], (img[0].shape[1], img[0].shape[0]))
		grds[img_name] = np.uint8(255 * grds[img_name])
		grds[img_name] = cv2.applyColorMap(grds[img_name], cv2.COLORMAP_JET)
		grds[img_name] = grds[img_name].astype(float)/255

def alpha_grad(grds, img):
	for img_name  in grds:
		grds[img_name] = grds[img_name] * 0.75 + img * 0.5

def create_image_grid(grds, grid):
	final = np.array([])
	for group in grid:
		line = np.array([])
		for img_name in group:
			image =  grds[img_name]
			if line.size == 0:
				line = image
			else:
				line = np.concatenate((line,image), axis=1)
	
		if final.size == 0:
			final = line
		else:
			final = np.concatenate((final,line), axis=0)
	return final

def get_args():
 
	parser = argparse.ArgumentParser(description='SIBGRAPI18 - Gender Activation Visualization\n')
	parser.add_argument('-m', metavar='--model', type=str, required='True', help='tensorflow model in .h5 format\n')
	parser.add_argument('-i', metavar='--image', type=str, help='input image\n')
	parser.add_argument('-v', metavar='--video', type=str, help='input video\n')
	parser.add_argument('-nw', metavar='--no_window', type=str, help='the code will not preset windoes, is recomended the usage of the -s option if you use -nw\n')
	parser.add_argument('-s', metavar='--save',  type=str, help='location to save the result with file name \n')

	args = parser.parse_args()
	h5_model	= args.m
	input_image = args.i
	input_video = args.v
	save_at	 = args.s
	no_window = rgs.nw

	video_mode = False
	input_location = input_image
	
	if no_window is None: 
		no_window = False
	else:
		no_window = True


	if input_video is not None:
		video_mode = True
		input_location = input_video
	elif input_image is None:
		raise InputError('You need to inform a video or an image input') from error

	return [h5_model, input_location, video_mode, save_at, no_window]

def _main():		
	[h5_model, input_location, video_mode, save_at, no_window] = get_args()

	# print("Parameters: ", h5_model, input_location, video_mode, save_at)

	colors = {'man': (0.9019, 0.7647, 0.6235), 
			  'woman': (0.6470, 0.3568, 1.0)}

	if not video_mode:
		norm_grup_detections = ["man","woman"]
		grid_detections = [["original","detection"], ["man","woman"]]
				 

		#load the tensorflow model, i am assuming that this code will be used to  gender detection cases, therefore, I hardcoded some data =p
		yolo_model, anchors, class_names = loadYoloModel(h5_model, 'cfg/anchors.txt', 'cfg/gender_classes.txt')


		model_image_size = yolo_model.layers[0].input_shape[1:3]
		img, img_size = loadAndNormalizeImg(input_location, model_image_size)

		sess = K.get_session() 

		yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
		input_image_shape = K.placeholder(shape=(2, ))

		boxes, scores, classes, boxes_scores = yolo_eval(yolo_outputs,
														input_image_shape,
														score_threshold=0.3,
														iou_threshold=0.5,
														grad_info=True)


		out_boxes, out_scores, out_classes, out_boxes_scores = sess.run(
													[boxes, scores, classes, boxes_scores],
													feed_dict={
													yolo_model.input: img,
													input_image_shape: [img.shape[1], img.shape[2]],
													K.learning_phase(): 0
													})
		
		print('Found {} boxes'.format(len(out_boxes)))

		# convert detections to my representatio of 
		detections = convertDetectionsToMyImp(out_boxes, out_scores, out_classes, class_names, img.shape)



		# create image with all detected bounding boxes
		imgAllDetections=img[0].copy()
		for d in detections:
			d.drawInImage(imgAllDetections, scale = 4, color=colors[d.classId], lineWidth=3, text=d.classId, alpha = 0.7)


		##TODO -> work with more than one detection
		for i, d in enumerate(detections):
			#finds were the detection was generated n the region tensor
			l = np.where(out_boxes_scores == out_scores[i])

			grds = {}
			grds["man"]		= heat_map(yolo_outputs[3][l[0][0],l[1][0],l[2][0],l[3][0],0], img, yolo_model)
			grds["woman"]	  = heat_map(yolo_outputs[3][l[0][0],l[1][0],l[2][0],l[3][0],1], img, yolo_model)

			normalize_grads(grds, norm_grup_detections)

			detectionImg = img[0].copy()

			d.drawInImage(detectionImg, scale = 4, color=colors[d.classId], lineWidth=3, text=d.classId, alpha = 0.7)

			alpha_grad(grds, detectionImg)

			grds["detection"] = detectionImg
			grds["original"] = img[0].copy()
	
			detectionGrad = create_image_grid(grds, grid_detections)

		detectionGrad = image_resize(detectionGrad, height = img_size[0], width = img_size[1], inter = cv2.INTER_AREA)
		
		if not no_window
			cv2.imshow("SIBGRAPI18 - Gender Detection", detectionGrad)
			key = cv2.waitKey(0)

		if save_at is not None:
			cv2.imwrite(save_at, detectionGrad)
			   
		sess.close()
		K.clear_session() 

if __name__ == '__main__':
    _main()

