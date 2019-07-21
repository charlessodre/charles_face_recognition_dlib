# Author: charlessodre
# Github: https://github.com/charlessodre/
# Create: 2019/07
# Note: This program is responsible for evaluating the performance of the model trained by the "face_recognition_dlib_train.py".

import cv2
import os
import dlib
import numpy as np
import time
import helper


def points_print(image_obj, points_face):
    for p in points_face.parts():
        cv2.circle(image_obj, (p.x, p.y), 2, (51, 255, 187), 1)

path_dlib_resources = "resources/dlib_resources/"

resources_path = "resources"

path_files = "source_img/test/"

output_path = "source_img/output/"

file_extension = "*.jpg"

total_detected_faces = 0

total_correctly_classified_images = 0

total_not_classified_images = 0

images_classified_wrong = {}

five_descriptors = True

if five_descriptors:
    predictor_shape_file = "shape_predictor_5_face_landmarks.dat"
else:
    predictor_shape_file = "shape_predictor_68_face_landmarks.dat"

points_detector = dlib.shape_predictor(os.path.join(path_dlib_resources, predictor_shape_file))

face_recognition = dlib.face_recognition_model_v1(
    os.path.join(path_dlib_resources, "dlib_face_recognition_resnet_model_v1.dat"))

if five_descriptors:
    resource_file_descriptors = "image_descriptors_05.npy"
    resource_file_descriptors_indexes = "image_descriptors_indexes_05.pickle"
else:
    resource_file_descriptors = "image_descriptors_68.npy"
    resource_file_descriptors_indexes = "image_descriptors_indexes_68.pickle"

file_faces_descriptors = np.load(os.path.join(resources_path, resource_file_descriptors))

faces_descriptors_file_indexes = np.load(os.path.join(resources_path, resource_file_descriptors_indexes))

debug_show_all_image_window = True
debug_show_rate_confidence_image = True
debug_show_image_name = True
debug_log_classifier_error = False
debug_show_rate_KNN_distance = True
debug_print_log_KNN_distance = False
debug_show_image_window_error_classifier = False
debug_print_log_faces_count = False
debug_save_image_classified = True


begin_test = time.time()
print(" ----------------- Begin Test: {} -----------------".format(helper.get_current_hour()))

list_images = helper.get_files_all_dirs(path_files, file_extension)

if len(list_images) == 0:
    print("No images found!")
    exit()

subdetector = ["Look Forward", "Left View", "Right View", "The front turning the left", "The front turning the right"]

for image_file in list_images:

    correctly_classified_image = False

    current_file_name = image_file.split('/')[-1]

    current_file_image_name = current_file_name.split('_')[0]

    image = cv2.imread(image_file)

    detector = dlib.get_frontal_face_detector()

    faces, confidences, idx = detector.run(image, 0, 0)

    # loop through detected faces
    i = 0
    for face, conf in zip(faces, confidences):

        e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))

        print("Detect: {}, Confidence: {}, Sub-detector: {}".format(face, confidences[i], subdetector[idx[i]]))
        i += 1

        cv2.rectangle(image, (e, t), (d, b), (0, 0, 255), 2)

        total_detected_faces += 1

        points = points_detector(image, face)

        points_print(image, points)

        face_descriptors = face_recognition.compute_face_descriptor(image, points)

        face_descriptor_list = [fd for fd in face_descriptors]

        np_array_face_descriptor = np.asanyarray(face_descriptor_list, dtype=np.float64)

        np_array_face_descriptor = np_array_face_descriptor[np.newaxis, :]

        distances = np.linalg.norm(np_array_face_descriptor - file_faces_descriptors, axis=1)

        min_distance_index = np.argmin(distances)

        min_distance_value = distances[min_distance_index]

        descriptor_file_name = os.path.split(faces_descriptors_file_indexes[min_distance_index])[1]

        descriptor_file_image_name = descriptor_file_name.split('_')[0]

        if debug_print_log_KNN_distance:
            print("Minimum KNN Distance: {:.2f}. Current Image: {}. Descriptor Image: {} ".format(min_distance_value,
                                                                                                  current_file_name,
                                                                                                  descriptor_file_name))

        text_image_classified = 'unknown'

        if current_file_image_name == descriptor_file_image_name:
            correctly_classified_image = True
            total_correctly_classified_images += 1
            text_image_classified = descriptor_file_image_name

        elif text_image_classified == 'unknown':

            total_not_classified_images += 1
        else:
            images_classified_wrong[image_file] = [current_file_image_name, descriptor_file_image_name]

        if debug_show_rate_confidence_image:
            cv2.putText(image, "conf {:.2f}".format(conf), (d + 5, t - 7), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        0.7,
                        (0, 0, 255))

        if debug_show_rate_KNN_distance:
            cv2.putText(image, "knn {:.2f}".format(min_distance_value), (d + 5, t + 25),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7,
                        (255, 255, 0))

        if debug_show_image_name:
            cv2.putText(image, text_image_classified, (d + 5, t + 11), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                        (0, 255, 0))

    detected_faces = len(faces)

    if debug_print_log_faces_count and detected_faces > 1:
        print("Detected Faces in image: Faces [{}]. File: {}".format(detected_faces, image_file))

    if debug_print_log_faces_count and detected_faces == 0:
        print("No detected Faces in image. File: {}".format(image_file))

    if debug_show_image_window_error_classifier and not correctly_classified_image:
        cv2.imshow("Image was not correctly classified", image)
        cv2.waitKey()

    if debug_save_image_classified:
        cv2.imwrite(output_path + current_file_name, image)

    if debug_show_all_image_window:
        cv2.imshow("Face Recognized", image)
        cv2.waitKey()


print("\n----------------------------- Classification Report -----------------------------\n")

print("Total processed images: {}.".format(len(list_images)))
print("Total detected faces: {}.".format(total_detected_faces))
print("Percentage detected faces: {}%.".format(total_detected_faces / len(list_images) * 100))
print("Total correctly classified faces: {}.".format(total_correctly_classified_images))
print("Total wrong classified faces: {}.".format(len(images_classified_wrong)))
print("Total not classified faces: {}.".format(total_not_classified_images))
print("Percentage correctly classified faces: {:.2f}%.".format(
    total_correctly_classified_images / total_detected_faces * 100))

print("Overall performance: {:.2f}%.".format(total_correctly_classified_images / len(list_images) * 100))

print("\n-------------------------------------------------------------------------------------")

print("\n----------------------------- Classified Error  -----------------------------\n")

for k, v in images_classified_wrong.items():
    print("image name real and classified: {} | image file: {}".format(v, k))

print("\n-------------------------------------------------------------------------------------")

end_test = time.time()
print("----------------- End Test: {} -----------------".format(helper.get_current_hour()))

print(
    "----------------- Time elapsed: {} -----------------".format(helper.format_seconds_hhmmss(end_test - begin_test)))

cv2.destroyAllWindows()
