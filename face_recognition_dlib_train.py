# Author: charlessodre
# Github: https://github.com/charlessodre/
# Create: 2019/07
# Info: This program is for the training to extract the main characteristics of the images. This program generates the files with the descriptors of the images.
# Note: Based in course "Reconhecimento de Faces e de Objetos com Python e Dlib" of Jones Granatyr (https://iaexpert.com.br/) in Udemy


import cv2
import os
import dlib
import numpy as np
import _pickle as pickle
import time
import helper


def points_print(image_obj, points_face):
    for p in points_face.parts():
        cv2.circle(image_obj, (p.x, p.y), 2, (51, 255, 187), 1)


path_dlib_resources = "resources/dlib_resources/"

path_files = "source_img/train"

file_extension = "*.jpg"

resources_path = "resources"

index_dict = {}

index = 0

faces_descriptor = None

total_detected_faces = 0

five_descriptors = True

if five_descriptors:
    resource_file_descriptors = "image_descriptors_05.npy"
    resource_file_descriptors_indexes = "image_descriptors_indexes_05.pickle"
else:
    resource_file_descriptors = "image_descriptors_68.npy"
    resource_file_descriptors_indexes = "image_descriptors_indexes_68.pickle"

if five_descriptors:
    predictor_shape_file = "shape_predictor_5_face_landmarks.dat"
else:
    predictor_shape_file = "shape_predictor_68_face_landmarks.dat"

points_detector = dlib.shape_predictor(os.path.join(path_dlib_resources, predictor_shape_file))

face_recognition = dlib.face_recognition_model_v1(
    os.path.join(path_dlib_resources, "dlib_face_recognition_resnet_model_v1.dat"))

begin_train = time.time()

print(" ----------------- Begin Train: {} -----------------".format(helper.get_current_hour()))

list_images = helper.get_files_all_dirs(path_files, file_extension)

if len(list_images) == 0:
    exit('No images Found!')

for image_file in list_images:

    image = cv2.imread(image_file)

    detector = dlib.get_frontal_face_detector()
    faces, confidences, idx = detector.run(image, 1, 1)

    faces_detect = detector(image, 1)

    total_detected_faces = len(faces)

    if total_detected_faces == 1:

        for face in faces_detect:

            points = points_detector(image, face)

            points_print(image, points)

            face_descriptors = face_recognition.compute_face_descriptor(image, points)

            print("\n--------- Image file: {} ---------".format(image_file))
            print("Image Points Detected: {}".format(len(face_descriptors)))

            face_descriptor_list = [fd for fd in face_descriptors]

            np_array_face_descriptor = np.asanyarray(face_descriptor_list, dtype=np.float64)

            np_array_face_descriptor = np_array_face_descriptor[np.newaxis, :]

            if faces_descriptor is None:
                faces_descriptor = np_array_face_descriptor

            else:
                faces_descriptor = np.concatenate((faces_descriptor, np_array_face_descriptor), axis=0)

            index_dict[index] = image_file

            index += 1


    elif total_detected_faces > 1:
        print("More than one face was detected in image: Faces [{}]. File: {}".format(total_detected_faces,
                                                                                      image_file))
        helper.move_file(image_file, 'source_img/test/')
    else:
        print("No face was detected in image. File: {}".format(image_file))
        helper.move_file(image_file, 'source_img/no_face/')

        np.save(os.path.join(resources_path, resource_file_descriptors), faces_descriptor)

        with open(os.path.join(resources_path, resource_file_descriptors_indexes), 'wb') as f:
            pickle.dump(index_dict, f)

        print(" ----------------- Info Train -----------------")
        print("Images Analysed: {}".format(len(list_images)))
        print("Image descriptor training file length: {}. Shape: {}".format(len(faces_descriptor),
                                                                            faces_descriptor.shape))

        end_train = time.time()
        print("\n----------------- End Train: {} -----------------".format(helper.get_current_hour()))

        print("----------------- Time elapsed: {} -----------------".format(
            helper.format_seconds_hhmmss(end_train - begin_train)))

        cv2.destroyAllWindows()
