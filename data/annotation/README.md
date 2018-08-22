# In the wild Gender Annotations

Details about how we anotate this data is discribed in our paper "Real-Time Gender Detection in the Wild Using Deep Neural Networks". Also, if you use our data cite the following paper:

@INPROCEEDINGS{zeni2018a,
	author={L. F. Zeni and C. R. Jung}, 
	booktitle={31st Conference on Graphics, Patterns and Images (SIBGRAPI 2018)}, 
	title={Real-Time Gender Detection in the Wild Using Deep Neural Networks}, 
	year={2018}, 
	pages={}, 
	keywords={gender detection, deep learning, visualization}, 
	doi={}, 
	month={Oct},}

## Datasets

We annotated two datasets to train our model that is able to deal with gender detection in the wild. AS we don't own the datasets rights we will only provide the annotations. All files are avaliable in this folder of our git repository.

## Pascal VOC 2007
	We have 4 diferent tar.gz files:

	> voc2007_lists_train_val.tar.gz
	Contain the lists of files used to train  and to evaluate our models

	> voc2007_xml.tar.gz
	Annotations in xml pascal VOC 2007 format.
	
	> voc2007_labels_darknet_format.tar.gz
	Annotations in darknet format (i.e.,  bounding boxes are normalizated by the size of the image).

	> voc2007_labels.tar.gz
	Annotations in txt format without normalization.

	## Setting Up

	We want a structure like this:

	VOCGender
		\___VOC2007
			\___JPEGImages/
			\___labels/
			\___train.txt
			\___val.txt


	1- Download Pascal VOC 2007 dataset images from 
	here: https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
	and here: https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar

	2- Extract the datasets and maintain only the folder "JPEGImages" inside it

	3- Download voc2007_labels_darknet_format.tar.gz and extract it inside the VOC2007 folder

	4- Download voc2007_lists_train_val.tar.gz and extract it inside the VOC2007 folder

	Done =)	


## CelebA
	We have 3 diferent tar.gz files:

	> celeba_lists_train_val_test.tar.gz
	Contain the lists of files used to train  and to evaluate our models
	
	> celeba_labels_darknet_format.tar.gz	
	Annotations in darknet format (i.e.,  bounding boxes are normalizated by the size of the image).

	> celeba_labels.tar.gz
	Annotations in txt format without normalization.

	## Setting Up

	We want a structure like this:

	CelebA
		\___imgs/
		\___labels/
		\___train.txt
		\___test.txt
		\___val.txt


	1- Download all CelebA images from here: https://drive.google.com/drive/folders/0B7EVK8r0v71peklHb0pGdDl6R28 (yeah, there are 14 parts of 7z files)

	2- Extract the images in the folder with name "img" 

	3- Download celeba_labels_darknet_format.tar.gz and extract it inside the CelebA folder

	4- Download celeba_lists_train_val_test.tar.gz and extract it inside the CelebA folder

	Done =)