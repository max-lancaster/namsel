@echo off

cd install_windows

SET addPath=c:\Python27;c:\gtk\bin;c:\Program Files\Scan Tailor

Rem Python 2.7 - 64bit
echo Installing Python 2.7...
msiexec /i python-2.7.14.amd64.msi /passive TARGETDIR=c:\Python27

Rem VC
echo Installing Microsoft Visual C++ Compiler...
msiexec /i VCForPython27.msi /passive
copy /Y msvc9compiler.py C:\Python27\Lib\distutils\msvc9compiler.py

Rem Scantailor
echo Installing Scantailor...
unzip -o scantailor.zip -d "c:\Program Files\Scan Tailor"

Rem GTK
echo Installing GTK+ Bundle...
unzip -o gtk+-bundle_2.22.1-20101229_win64.zip -d c:\gtk

	Rem pycairo
	echo Installing pycairo...
	unzip -o py2cairo-1.10.0.win-amd64-py2.7.zip -d c:\Python27

	Rem pygtk
	echo Installing pygtk...
	unzip -o pygtk-2.22.0.win-amd64-py2.7.zip -d c:\Python27
		
	Rem pygobject
	echo Installing pygobject...
	unzip -o pygobject-2.28.6.win-amd64-py2.7.zip -d c:\Python27

setX /M PATH "%PATH%;%addPath%"
SET PATH=%PATH%;%addPath%

rem More stuffs
echo Installing pip packages...
cd packages
c:\Python27\python -m pip install Cython-0.27.3-cp27-cp27m-win_amd64.whl numpy-1.14.1-cp27-none-win_amd64.whl opencv_contrib_python-3.4.0.12-cp27-cp27m-win_amd64.whl Pillow-5.0.0-cp27-cp27m-win_amd64.whl scikit_learn-0.18.1-cp27-cp27m-win_amd64.whl scipy-1.0.0-cp27-none-win_amd64.whl simplejson-3.13.2-cp27-cp27m-win_amd64.whl sklearn-0.0.tar.gz
cd ..

Rem Fonts installation
echo Installing the fonts
mkdir \tmp
unzip ..\data_generation\fonts.zip -d \tmp\tibetan-fonts
cscript install-fonts.vbs
rmdir \tmp\tibetan-fonts /s /q

Rem Building Namsel
echo Building Namsel...
cd ..
c:\Python27\python setup.py build_ext --inplace

echo Generating the datas...
cd data_generation
c:\Python27\python font_draw.py

cd ..\datasets
..\install_windows\unzip datapickles.zip

cd ..
c:\Python27\python classify.py

echo. & echo.All Done! & echo. & echo.
