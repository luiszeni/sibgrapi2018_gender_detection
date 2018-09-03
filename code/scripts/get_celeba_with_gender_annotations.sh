#!/bin/bash

# #dependecies
# pip install requests
# pip install tqdm
# sudo apt-get install p7zip-full

git clone https://github.com/chentinghao/download_google_drive.git

echo "downloading file 1 of 14"
python download_google_drive/download_gdrive.py 0B7EVK8r0v71pQy1YUGtHeUM2dUE img_celeba.7z.001
echo "downloading file 2 of 14"
python download_google_drive/download_gdrive.py 0B7EVK8r0v71peFphOHpxODd5SjQ img_celeba.7z.002
echo "downloading file 3 of 14"
python download_google_drive/download_gdrive.py 0B7EVK8r0v71pMk5FeXRlOXcxVVU img_celeba.7z.003
echo "downloading file 4 of 14"
python download_google_drive/download_gdrive.py 0B7EVK8r0v71peXc4WldxZGFUbk0 img_celeba.7z.004
echo "downloading file 5 of 14"
python download_google_drive/download_gdrive.py 0B7EVK8r0v71pMktaV1hjZUJhLWM img_celeba.7z.005
echo "downloading file 6 of 14"
python download_google_drive/download_gdrive.py 0B7EVK8r0v71pbWFfbGRDOVZxOUU img_celeba.7z.006
echo "downloading file 7 of 14"
python download_google_drive/download_gdrive.py 0B7EVK8r0v71pQlZrOENSOUhkQ3c img_celeba.7z.007
echo "downloading file 8 of 14"
python download_google_drive/download_gdrive.py 0B7EVK8r0v71pLVltX2F6dzVwT0E img_celeba.7z.008
echo "downloading file 9 of 14"
python download_google_drive/download_gdrive.py 0B7EVK8r0v71pVlg5SmtLa1ZiU0k img_celeba.7z.009
echo "downloading file 10 of 14"
python download_google_drive/download_gdrive.py 0B7EVK8r0v71pa09rcFF4THRmSFU img_celeba.7z.010
echo "downloading file 11 of 14"
python download_google_drive/download_gdrive.py 0B7EVK8r0v71pNU9BZVBEMF9KN28 img_celeba.7z.011
echo "downloading file 12 of 14"
python download_google_drive/download_gdrive.py 0B7EVK8r0v71pTVd3R2NpQ0FHaGM img_celeba.7z.012
echo "downloading file 13 of 14"
python download_google_drive/download_gdrive.py 0B7EVK8r0v71paXBad2lfSzlzSlk img_celeba.7z.013
echo "downloading file 14 of 14"
python download_google_drive/download_gdrive.py 0B7EVK8r0v71pcTFwT1VFZzkzZk0 img_celeba.7z.014

7z x img_celeba.7z.001

mkdir ../../data/CelebA
mv img_celeba ../../data/CelebA/img
cd ../../data/CelebA

tar -xvzf ../annotation/celeba_labels_darknet_format.tar.gz
tar -xvzf ../annotation/celeba_lists_train_val_test.tar.gz

#change to right absolute path to annotations
DATA_LOCATION=$('pwd') 
ORIGINAL_LOC='/home/zeni/projects/in_progress/sibgraphi_2018/data/CelebA'

sed -i "s|$ORIGINAL_LOC|$DATA_LOCATION|" test.txt
sed -i "s|$ORIGINAL_LOC|$DATA_LOCATION|" train.txt
sed -i "s|$ORIGINAL_LOC|$DATA_LOCATION|" val.txt 