# SIBGRAPI 2018 -  Real Time Gender Detection in the Wild #
This repo contains the source code to reproduce our results of our paper published in the SIBGRAPHI2018. Our code is projected to run on Linux environments.

More details @  http://luiszeni.com.br/gender_sib2018/

If you want an easy way to test and reproduce our method, check out our docker image @  Docker Hub: [link]

## This project is organized in the following manner:
sibgrapi2018_gender_detection<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|___code<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|___darknet 
			**[Darknet framework used to train and run our models]**<br>			
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|___scripts 
			**[Scripts to download the datasets with our annotated data]**<br>		
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|___visualization 
			 **[Heatmap activation tool to visualize activations for each class of an trained model]**<br>	 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|___data<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|___annotations 
			**[Set of annotations used to train our models]**<br>		
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|___img 
			**[Testing images]**<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|___vid 
			**[Testing videos]**<br>

## Running the gender detector on darknet
	
1- Build the project (see the makefile flags if you are not using a GPU or openCV, more details at darknet site)
```
cd code/darknet
make
```
2- Download our pre-trained model:
```
wget http://inf.ufrgs.br/~lfazeni/sib2018_models/gender_detection_50voc_50celeb_darknet.weights
```
3- Run the demo
```
./darknet detector demo cfg/test_voc_only.data cfg/yoloGender.cfg gender_detection_50voc_50celeb_darknet.weights ../../data/vid/001.mp4
```
## Training a gender detector model on darknet with celebA and PascalVoc
1- Download the datasets running the scripts in c
```
cd code/scripts
./get_celeba_with_gender_annotations.sh
./get_voc_with_gender_annotations.sh
```
2- download yolo v2 pre-trained mode and cfg:
```
cd ../../
cd code/darknet
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2-voc.cfg
wget https://pjreddie.com/media/files/yolov2.weights
```
3- get only the 29 first layers of the model
```
./darknet partial yolov2-voc.cfg yolov2.weights yolov2.weights.29 29
```
4- Train (this code trains our model using in each epoch 50% of image from each dataset)
```
mkdir backup
mkdir backup/gender_voc_50_celeb_50
./darknet detector train cfg/train_voc_50_celeb_50.data cfg/yoloGender.cfg yolov2.weights.29 29
```

## Visualizing the heatmap actvations
Dependencies:  tensorflow, keras, opencv in python, numpy

1- Downloading the tensorflow's model 
```
cd code/visualization
wget http://inf.ufrgs.br/~lfazeni/sib2018_models/gender_detection_50voc_50celeb_darknet.weights
```
2- Visualizing the heatmap of an image:
```
python heatmap_from_detection.py -m gender_detection_50voc_50celeb_tensorflow.h5  -i ../../data/img/000058.jpg -md 3
```
3- Visualizing the heatmap of an video:
```
python heatmap_from_detection.py -m gender_detection_50voc_50celeb_tensorflow.h5  -v ../../data/vid/001.mp4 -md 2
```
4- View all avaible options
```
python heatmap_from_detection.py --help
```

