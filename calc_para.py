# -*- coding: utf-8 -*-

import argparse
import pickle
import numpy as np
import cv2


def parse_arg():
    parser = argparse.ArgumentParser(description='Calc calibration parameters from chess-board or circles-gird images.')
    parser.add_argument('-s', '--size', type=float, default=18.5, help='the size of a grid in mm.')
    parser.add_argument('-x', '--x_num', type=int, default=9, help='the horizontal number of grids.')
    parser.add_argument('-y', '--y_num', type=int, default=9, help='the vertical number of grids.')
    parser.add_argument('-p', '--parameter_file', type=str, default='param.pickle',
                        help='the outout file name for the parameters.')
    parser.add_argument('-c', '--use_circle_grid', action='store_true',
                        help='use circles-grid images instead of chess-board images.')
    parser.add_argument('ref_images', type=str, nargs='+', help='a list of chess-board or circles-grid images.')
    return parser.parse_args()


def prepare_a_matrix_of_pattern_points(grid_size, grid_intersection_size):
    pattern_points = np.zeros((np.prod(grid_intersection_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(grid_intersection_size).T.reshape(-1, 2)
    pattern_points *= grid_size
    return pattern_points


def read_circle_grid_images(ref_images, grid_intersection_size):
    image_shape = None
    image_points = []
    for circle_grid_image in ref_images:
        print(circle_grid_image)
        image = cv2.imread(circle_grid_image, cv2.IMREAD_GRAYSCALE)
        if image_shape is None:
            image_shape = (image.shape[1], image.shape[0])
        found, corner = cv2.findCirclesGrid(image, grid_intersection_size,
                                            flags=cv2.CALIB_CB_SYMMETRIC_GRID+cv2.CALIB_CB_CLUSTERING)
        if found:
            print("findCirclesGrid: Success.")
            image_points.append(corner)
        else:
            print("findCirclesGrid: Failure.")
    return image_shape, image_points


def read_chess_images(ref_images, grid_intersection_size):
    image_shape = None
    image_points = []
    for chess_image in ref_images:
        print(chess_image)
        image = cv2.imread(chess_image, cv2.IMREAD_GRAYSCALE)
        if image_shape is None:
            image_shape = (image.shape[1], image.shape[0])
        found, corner = cv2.findChessboardCorners(image, grid_intersection_size)
        if found:
            print("findChessboardCorners: Success.")
            image_points.append(corner)
        else:
            print("findChessboardCorners: Failure.")
    return image_shape, image_points


def calc_parameters(square_side_length, grid_intersection_size, image_points, image_shape):
    pattern_points = prepare_a_matrix_of_pattern_points(square_side_length, grid_intersection_size)
    object_points = [pattern_points] * len(image_points)
    return cv2.calibrateCamera(object_points, image_points, image_shape, None, None)


def evaluate_parameters(K, d, r, t, square_side_length, grid_intersection_size, image_points):
    pattern_points = prepare_a_matrix_of_pattern_points(square_side_length, grid_intersection_size)
    object_points = [pattern_points] * len(image_points)

    mean_error = 0
    for i in range(len(object_points)):
        image_points2, _ = cv2.projectPoints(object_points[i], r[i], t[i], K, d)
        error = cv2.norm(image_points[i], image_points2, cv2.NORM_L2) / len(image_points2)
        mean_error += error
    return mean_error / len(object_points)


def main():
    args = parse_arg()

    grid_intersection_size = (args.x_num, args.y_num)
    if args.use_circle_grid:
        image_shape, image_points = read_circle_grid_images(args.ref_images, grid_intersection_size)
    else:
        image_shape, image_points = read_chess_images(args.ref_images, grid_intersection_size)
    if 0 < len(image_points):
        rms, K, d, r, t = calc_parameters(args.size, grid_intersection_size, image_points, image_shape)
        err = evaluate_parameters(K, d, r, t, args.size, grid_intersection_size, image_points)
        print("K = \n", K)
        print("d = {}".format(d.ravel()))
        print("total error: {}".format(err))

        # save parameters
        parameters = {'K': K, 'd': d}
        with open(args.parameter_file, 'wb') as f:
            pickle.dump(parameters, f)
    else:
        print('Even one grid could not be found.')


if __name__ == '__main__':
    main()
