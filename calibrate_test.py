# -*- coding: utf-8 -*-

import os.path
import argparse
import pickle
import numpy as np
import cv2


def parse_arg():
    parser = argparse.ArgumentParser(description='Calibiration test for fisheye images.')
    parser.add_argument('-p', '--parameter_file', type=str, nargs=1, help='the name of the parameter file.')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='the directory name where calibrated images will store to.')
    parser.add_argument('fisheye_images', type=str, nargs='+', help='a list of fisheye images.')
    return parser.parse_args()


def main():
    args = parse_arg()

    with open(args.parameter_file[0], 'rb') as f:
        parameters = pickle.load(f)
    print("K = \n", parameters['K'])
    print("d = {}".format(parameters['d'].ravel()))

    camera_mat = parameters['K']
    dist_coef = parameters['d']
    for fisheye_image in args.fisheye_images:
        image = cv2.imread(fisheye_image)
        undistort_image = cv2.undistort(image, camera_mat, dist_coef)

        if args.output_dir is not None:
            basename = os.path.basename(fisheye_image)
            output_file = '{}/{}'.format(args.output_dir, basename)
            cv2.imwrite(output_file, undistort_image)
            print("{} ---> {} ".format(fisheye_image, output_file))


if __name__ == '__main__':
    main()
