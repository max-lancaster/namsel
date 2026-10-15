#! /usr/bin/env bash
# Do all of this and then save a snapshot

sudo apt-get install python-pip python2.7-dev \
                     build-essential python-setuptools \
                     libatlas-dev libatlas3-base ghostscript scantailor \
                     python-opencv libtiff-tools python-gtk2 python-cairo unzip

sudo pip2 install Cython pillow sklearn requests simplejson wheel scipy numpy
sudo pip2 install --upgrade scikit-learn==0.18.1

python setup.py build_ext --inplace

cd data_generation
mkdir -p ~/.fontscp ~/.fonts/
unzip fonts.zip -d ~/.fonts
fc-cache -f -v
python font_draw.py

cd ../datasets
unzip datapickles.zip

cd ..
python classify.py

echo -e '\nAll Done!\n\n'