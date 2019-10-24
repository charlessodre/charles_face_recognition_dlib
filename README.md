Computational Vision - Studies on face detection and facial recognition with Python. This study uses the dlib and opencv libraries to detect and classify faces by assigning a name.

cvlib : https://github.com/arunponnusamy/cvlib
opencv: https://docs.opencv.org/master/d6/d00/tutorial_py_root.html
dlib: http://dlib.net/

-------------------------------------------------------------------------------------------------------------

Information about "visao_compu.yml " file:

visao_compu.yml -  Information about the packages and versions used in the programs.


-------------------------------------------------------------------------------------------------------------
Information about ".py" files:

face_recognition_dlib_train.py – program responsible for the training to extract the main characteristics of the images. This program generates the files with the descriptors of the images.

face_recognition_dlib_test.py – program responsible for evaluating the performance of the model trained by the "face_recognition_dlib_train.py". This program provides some statistics about face recognition.

helper.py - Miscellaneous support functions.

-------------------------------------------------------------------------------------------------------------

Directory Information:

resources –  training files directory saved.

dlib_resources – files directory dlib.

test - images for model testing. These images SHOULD HAVE ONLY ONE FACE and the file name should start with face name in the format "facename_xxxxx.jpeg". The image extent can be changed in the program.

train - directory of images for model training. These images SHOULD ALSO HAVE ONLY ONE FACE and the file should start with face name in the format "facename_xxxxx_xxxxx.jpeg".
