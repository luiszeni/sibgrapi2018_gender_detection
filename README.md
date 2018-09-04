# SIBGRAPI 2018 -  Real Time Gender Detection in the Wild #
This repo contains the source code to reproduce our results of our paper published in the SIBGRAPHI2018. Our code is projected to run on Linux environment.

More details @  http://luiszeni.com.br/gender_sib2018/

## This project is organized in the following manner:

	gender_sib2018/
	\___code/
		\___darknet/
			-->darknet framework used to train our model.
		\___scripts/
			--> scripts to download the datasets with our annotated data.s
		\___visualization/
			--> our heatmap activation tool to visualize the activations of the network.
	\___data/
		\___annotations
			--> set of annotations used to train our models
		\___img
			--> testing images

## Running the gender detector on darknet
	
	1- build the project (see the makefile flags if you are not using a GPU or openCV, more details at darknet site)
		cd code/darknet
		make
	2- download our pre-trained model:
		wget http://inf.ufrgs.br/~lfazeni/sib2018_models/gender_detection_50voc_50celeb_darknet.weights

	3- run the demo
		./darknet detector demo cfg/test_voc_only.data cfg/yoloGender.cfg gender_detection_50voc_50celeb_darknet.weights

## Training a gender detector model on darknet with celebA and PascalVoc
	1- Download the datasets running the scripts in c
		cd code/scripts
		./get_celeba_with_gender_annotations.sh
		./get_voc_with_gender_annotations.sh

	2- download yolo v2 pre-trained mode and cfg:
		cd ../../
		cd code/darknet
		wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2-voc.cfg
		wget https://pjreddie.com/media/files/yolov2.weights

	3- get only the 29 first layers of the model
		./darknet partial yolov2-voc.cfg yolov2.weights yolov2.weights.29 29

	4- Train (this code trains our model using in each epoch 50% of image from each dataset)
		mkdir backup
		mkdir backup/gender_voc_50_celeb_50
		./darknet detector train cfg/train_voc_50_celeb_50.data cfg/yoloGender.cfg yolov2.weights.29 29


## Visualizing the heatmap actvations
	Our tool to visualize the activations of our model.
