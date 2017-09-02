#! /usr/bin/env bash
# Do all of this and then save a snapshot
sudo apt-get install python-pip python2.7-dev \
                     build-essential python-setuptools \
                     libatlas-dev libatlas3-base ghostscript scantailor \
                     python-opencv libtiff-tools python-gtk2 python-cairo unzip
sudo pip2 install Cython pillow sklearn requests simplejson wheel scipy numpy
python setup.py build_ext --inplace
mkdir -p ~/.fontscp data_generation/fonts/*ttf ~/.fonts/
fc-cache -f -v
cd data_generation
python font_draw.py
cd ..
cd datasets
unzip datapickles.zip
cd ..
python classify.py
