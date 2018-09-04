# SIBGRAPI 2018 -  Real Time Gender Detection in the Wild #
Intructions to run our docker image.

More details of our paper @  http://luiszeni.com.br/gender_sib2018/

## Dependecies:
nvidia-docker and a nvidia GPU.

## Instructions
	
1- Pull the image
```
docker pull luiszeni/sibgrapi18_gender_detection
```
2- Create an container from this image:
```
xhost +local:root; nvidia-docker run -ti --name sib18_gender_detec -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix 61a3ffffa35c  bash
```
This command allows you to run the code using the host DISPLAY (i.e.  you can see the windows generated inside the docker container in your machine). It also start the container in the bash script inside the project of our work (You shold see someting like this "(cv) sibgrapi2018_gender_detection#" ).

2.1- Starting the container again:
If you exit the container you can restart it using the following command:
```
xhost +local:root; nvidia-docker start -ai sib18_gender_detec
```
3- Testing the model in real-time
```
cd code/darknet
./darknet detector demo cfg/test_voc_only.data cfg/yoloGender.cfg gender_detection_50voc_50celeb_darknet.weights ../../data/vid/001.mp4
```

4- Training a gender detector model on darknet with celebA and PascalVoc
4.1- Download the datasets running the scripts to download
```
cd code/scripts
./get_celeba_with_gender_annotations.sh
./get_voc_with_gender_annotations.sh
```
4.2- download yolo v2 pre-trained mode and cfg:
```
cd ../../
cd code/darknet
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2-voc.cfg
wget https://pjreddie.com/media/files/yolov2.weights
```
4.3- get only the 29 first layers of the model
```
./darknet partial yolov2-voc.cfg yolov2.weights yolov2.weights.29 29
```
4.4- Train (this code trains our model using in each epoch 50% of image from each dataset)
```
mkdir backup
mkdir backup/gender_voc_50_celeb_50
./darknet detector train cfg/train_voc_50_celeb_50.data cfg/yoloGender.cfg yolov2.weights.29 29
```

5- Visualizing the heatmap actvations

5.1- Visualizing the heatmap of an image:
```
cd code/visualization
python heatmap_from_detection.py -m gender_detection_50voc_50celeb_tensorflow.h5  -i ../../data/img/000058.jpg -md 3
```
5.2- Visualizing the heatmap of an video:
```
cd code/visualization
python heatmap_from_detection.py -m gender_detection_50voc_50celeb_tensorflow.h5  -v ../../data/vid/001.mp4 -md 2
```
5.3- View all avaible options
```
python heatmap_from_detection.py --help
```


