#!/usr/bin/env python3
import os
import sys
import cv2 as cv


def main():
    if len(sys.argv) < 2:
        return

    input_image_folder = sys.argv[1]
    print("input image folder: ", input_image_folder)

    if not os.path.exists(input_image_folder) or not os.path.isdir(input_image_folder):
        print("Please input image directory!")
        return

    images = os.listdir(input_image_folder)
    print("len: ", len(images))

    # create output folder to store descriptors
    output_images_descriptors_folder = "output_images_descriptors"
    if not os.path.exists(output_images_descriptors_folder):
        os.mkdir(output_images_descriptors_folder)

    # Initiate ORB detector
    orb = cv.ORB_create()

    for image in images:
        splitext = os.path.splitext(image)
        if len(splitext) > 1:
            base_filename = splitext[0]
            file_extension = splitext[1]
            if file_extension == '.jpg' or \
                    file_extension == '.JPG' or \
                    file_extension == '.png' or \
                    file_extension == '.PNG':
                # read image mat
                img = cv.imread(os.path.join(input_image_folder, image))
                # find the keypoints with ORB
                kp = orb.detect(img, None)
                # compute the descriptors with ORB
                kp, des = orb.compute(img, kp)
                # store orb descriptors
                store_file_path = os.path.join(output_images_descriptors_folder, base_filename + ".orb")
                with open(store_file_path, "w") as text_file:
                    print(des, file=text_file)


if __name__ == '__main__':
    main()
