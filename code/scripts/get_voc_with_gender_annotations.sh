#!/bin/bash
#get voc from redmon's server (it is fastest =p)
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar

tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar

rm VOCtrainval_06-Nov-2007.tar
rm VOCtest_06-Nov-2007.tar

mv VOCdevkit ../../data/VOCGender
cd ../../data/VOCGender

# remove data that we don't need
rm -r VOC2007/Annotations/
rm -r VOC2007/ImageSets/
rm -r VOC2007/SegmentationClass/
rm -r VOC2007/SegmentationObject/

tar -xvzf ../annotation/voc2007_lists_train_val.tar.gz

cd VOC2007
tar -xvzf ../../annotation/voc2007_labels_darknet_format.tar.gz
tar -xvzf ../../annotation/voc2007_xml.tar.gz

cd ..

#change to right absolute path to annotations
DATA_LOCATION=$('pwd') 
ORIGINAL_LOC='/home/zeni/projects/in_progress/sibgraphi_2018/data/VOCGender'

sed -i "s|$ORIGINAL_LOC|$DATA_LOCATION|" 2007_train.txt
sed -i "s|$ORIGINAL_LOC|$DATA_LOCATION|" 2007_val.txt