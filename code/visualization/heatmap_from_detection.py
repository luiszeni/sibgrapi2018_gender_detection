import cv2, os, time, argparse
import numpy as np

from keras import backend as K

from yad2k.keras_yolo import yolo_eval, yolo_head, loadYoloModel

from core.BoundBox import BoundBox
from core.Utils import *

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

def normalize_grads(img, img_size):
		img /= np.max(img)
		img = cv2.resize(img, (img_size[1], img_size[0]))
		img = np.uint8(255 * img)
		img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
		return img.astype("float32")/255

def alpha_grad(img, orig_image, img_orig_size=None):
		cv2.addWeighted(img, 0.6, orig_image, 0.4, 0, img)
		if img_orig_size is not None:
			img = image_resize(img, height = img_orig_size[0], width = img_orig_size[1], inter = cv2.INTER_AREA)
		return img

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


def detection_heatmap_proceess_frame(img, h5_model, max_detections=3, old_detections=None):
	colors = {'man': (0.9019, 0.7647, 0.6235), 'woman': (0.6470, 0.3568, 1.0)}
	print("max_detections", max_detections)
	#load the tensorflow model, i am assuming that this code will be used to  gender detection cases, therefore, I hardcoded some data =p
	yolo_model, anchors, class_names = loadYoloModel(h5_model, 'cfg/anchors.txt', 'cfg/gender_classes.txt')

	model_image_size = yolo_model.layers[0].input_shape[1:3]

	img_orig_size = img.shape
	img  = normalize_img(img, model_image_size)

	sess = K.get_session() 

	yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
	input_image_shape = K.placeholder(shape=(2, ))

	boxes, scores, classes, boxes_scores, box_xy, box_wh, box_confidence, box_class_probs = yolo_eval(yolo_outputs,
													input_image_shape,
													score_threshold=0.3,
													iou_threshold=0.5,
													grad_info=True)

	out_boxes, out_scores, out_classes, out_boxes_scores, out_box_xy, out_box_wh, out_box_confidence, out_box_class_probs = sess.run(
												[boxes, scores, classes, boxes_scores, box_xy, box_wh, box_confidence, box_class_probs],
												feed_dict={
												yolo_model.input: img,
												input_image_shape: [img.shape[1], img.shape[2]],
												K.learning_phase(): 0
												})
	
	print('Found {} boxes'.format(len(out_boxes)))

	# convert detections to my representationnnn of bb 
	detections = convert_detections_to_my_imp(out_boxes, out_scores, out_classes, class_names, img.shape)
	detections = sorted(detections, key=lambda x: x.pred, reverse=True)
	detections = detections[:max_detections]

	images = []
	for i, d in enumerate(detections):
		#finds were the detection was generated n the region tensor
		l = np.where(out_boxes_scores == out_scores[i])
		grds = {}
		
		img_ind = l[0][0];
		x = l[1][0]
		y =  l[2][0]
		anchor = l[3][0]
		gender = l[4][0]


		detectionImg = img[0].copy()
		grds["heat"] = heat_map(yolo_model.output[ img_ind, x, y, anchor*7 + 5 + gender ], img, yolo_model)
		grds["heat"] = normalize_grads(grds["heat"], img[0].shape)
		grds["heat"] = alpha_grad(grds["heat"], detectionImg, img_orig_size)

		d.drawInImage(detectionImg, scale = 4, color=colors[d.classId], lineWidth=3, text=d.classId, alpha = 0.7)
		
		grds["detection"] = image_resize(detectionImg, height = img_orig_size[0], width = img_orig_size[1], inter = cv2.INTER_AREA)

		images.append(create_image_grid(grds,  [["detection", "heat"]]))


	images, detections = the_worst_tracking(detections[:], old_detections, images[:])
	for i in range(len(images), max_detections):
		if len(images) is 0:
			orig_resized = image_resize(img[0].copy(), height = img_orig_size[0], width = img_orig_size[1], inter = cv2.INTER_AREA)
			images.append( np.concatenate((orig_resized,orig_resized*0), axis=1))
		else:
			orig_resized = image_resize(img[0].copy(), height = img_orig_size[0], width = img_orig_size[1], inter = cv2.INTER_AREA)
			images.append( np.concatenate((orig_resized*0,orig_resized*0), axis=1))


	output_img = images[0]
	for i in range(1, len(images)):
		output_img =  np.concatenate((output_img,images[i]), axis=0)

	sess.close()
	K.clear_session() 

	return output_img, detections


def get_args():
	parser = argparse.ArgumentParser(description='SIBGRAPI18 - Gender Activation Visualization\n')
	parser.add_argument('-m',  metavar='--model',      type=str, required='True', help='tensorflow model in .h5 format\n')
	parser.add_argument('-i',  metavar='--image',      type=str, help='input image\n')
	parser.add_argument('-v',  metavar='--video',      type=str, help='input video\n')
	parser.add_argument('-nw', metavar='--no_window',  type=bool, default=False, help='the code will not preset windows, is recommended the usage of the -s option if you use -nw\n')
	parser.add_argument('-s',  metavar='--save',       type=str, help='full path to location to save the result with file name and file extension\n')

	parser.add_argument('-sf', metavar='--starting_frame',  type=int, default=0,  help='Video Mode: Frame from whith the application  starts processing\n')
	parser.add_argument('-mf', metavar='--max_frame',       type=int, default=-1, help='Video Mode: Amount of frames  whith the application processes agter the starting frame\n')
	parser.add_argument('-skf',metavar='--skip_frames',     type=int, default=1,  help='Video Mode: Number of frames to skip during processing (default is one)\n')
	parser.add_argument('-md', metavar='--max_detections',  type=int, default=2,  help='Video Mode: Number of detections which will be displayed (default is two)\n')

	args = parser.parse_args()
	h5_model	= args.m
	input_image = args.i
	input_video = args.v
	save_at	 = args.s
	no_window = args.nw

	starting_frame = args.sf
	max_frame = args.mf
	skip_frames = args.skf
	max_detections = args.md

	video_mode = False
	input_location = input_image

	if input_video is not None:
		video_mode = True
		input_location = input_video
	elif input_image is None:
		raise InputError('You need to inform a video or an image input') from error

	return {"h5_model":h5_model, "input_location":input_location, "video_mode":video_mode, 
			"save_at":save_at, "no_window":no_window, "starting_frame":starting_frame, 
			"max_frame":max_frame, "skip_frames":skip_frames, "max_detections":max_detections}

def _main():		
	args = get_args()
	if not args["video_mode"]:
		img = cv2.imread(args["input_location"])
		# output_img = detection_heatmap_proceess_frame(img, args["h5_model"])
		output_img, old_detections =  detection_heatmap_proceess_frame(img, args["h5_model"], args["max_detections"], [])

		if not args["no_window"]:
			to_show = image_resize(output_img, height = 800, inter = cv2.INTER_AREA)
			cv2.imshow("SIBGRAPI18 - Gender Detection", to_show)
			cv2.waitKey(0)

		if args["save_at"] is not None:
			output_img *= 255
			output_img = output_img.astype(np.uint8)
			cv2.imwrite(args["save_at"] , output_img)

	else:
		cap = cv2.VideoCapture(args["input_location"])
		print("args[max_detections]", args["max_detections"])
		frame_size = None
		if not cap.isOpened(): 
		   print("uops, not able to open the capture in opencv... =(")
		else:
			frame_size = int(cap.get(3)), int(cap.get(4))
			for f in range(0, args["starting_frame"]):
				ret, frame = cap.read()
				if ret is not True:
					break
				if not args["no_window"]:
					cv2.imshow('SIBGRAPI18 - Gender Detection',frame)
					if cv2.waitKey(1) == 27:
						break

			old_detections = None
			if args["save_at"] is not None:
				fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
				out = cv2.VideoWriter(args["save_at"],fourcc, 24.0, (frame_size[0]*args["max_detections"],frame_size[1]*2))
			
			f = 0
			while f < args["max_frame"] or args["max_frame"] is -1:
				ret, frame = cap.read()
				if ret is not True:
					break
				if f % args["skip_frames"] is 0:
					# Beer is life, don't you think?
					print("Processing Frame:", args["starting_frame"] + f)

					output_img, old_detections =  detection_heatmap_proceess_frame(frame, args["h5_model"], args["max_detections"], old_detections)

					output_img *= 255
					output_img = output_img.astype(np.uint8)	
					if args["save_at"] is not None:
						out.write(output_img)

					if not args["no_window"]:
						to_show = image_resize(output_img, height = 800, inter = cv2.INTER_AREA)
						cv2.imshow('SIBGRAPI18 - Gender Detection',to_show)
						if cv2.waitKey(1) == 27:
							break
				f+=1
			cap.release()
			cv2.destroyAllWindows()
			if args["save_at"] is not None:
				out.release()

if __name__ == '__main__':
	_main()

