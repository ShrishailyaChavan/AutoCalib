
import numpy as np
import cv2
import math
import argparse
from glob import glob
from scipy import optimize as opt
from typing import List
import os
import random

def convert_vector_to_matrix(x):
    alpha, gamma, beta, u, v = x

    A1 = [alpha, gamma, u]
    A2 = [0, beta, v]
    A3 = [0, 0, 1]

    A = np.vstack((A1, A2, A3))

    return A

#Plotting the reprojected/calibration points and original points on the dataset provided
def projection_from_3D_to_2D(img_pts, calibration_points, images):

    for i, (imagepath, image_points, reprojected_points) in enumerate(zip(images, img_pts, calibration_points)):
        image = cv2.imread(imagepath)

        for point, reprojected_point in zip(image_points, reprojected_points):
            x, y = int(point[0]), int(point[1])
            x_reprojected_point, y_reprojected_point = int(reprojected_point[0]), int(reprojected_point[1])
            cv2.circle(image, (x, y), 15, (0, 0, 255), 5)
            cv2.circle(image, (x,y), 3, (255, 0, 0), 2)
            
    
    
        cv2.imwrite("Output/{}.jpg".format(i), image)

    return image
     

def compute_extrinsic(K, homography_matrix):
    #we compute extrinsic parameters from 2.3 section of paper
    
    lmda = (1 / np.linalg.det(K))
    
    K_inverse = np.linalg.inv(K)

    h1 = homography_matrix[:, 0]
    h2 = homography_matrix[:, 1]
    h3 = homography_matrix[:, 2]

    r1 = np.dot(K_inverse, h1)
    r1 = r1/lmda

    r2 = np.dot(K_inverse, h2)
    r2 = r2/lmda

    t = np.dot(K_inverse, h3)/lmda

    r3 = np.cross(r1, r2)

    R = np.asarray([r1, r2, r3])
    R = R.T

    return R, t


def optimized_cost_function(intrinsic, img_pts, world_points, homography_matrices):
    A = np.zeros(shape=(3, 3))
    A[0, 0], A[1, 1], A[0, 2], A[1, 2], A[0, 1], A[2,2] = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3], intrinsic[4], 1
    #here we have k1, k2 as zero as our initial estimate so that camera has minimal distortion
    k1, k2 = intrinsic[5], intrinsic[6]
    u0, v0 = intrinsic[2], intrinsic[3]

    reprojection_error = np.empty(shape=(26*9*6), dtype=np.float64)
    i = 0
    for image_points, homography_matrix in zip(img_pts, homography_matrices):
        R, t = compute_extrinsic(A, homography_matrix)

        projection_matrix = np.zeros((3, 4))
        projection_matrix[:, :-1] = R
        projection_matrix[:, -1] = t

        for point, world_point in zip(image_points, world_points):
            M = np.array([[world_point[0]], [world_point[1]], [0], [1]])
            ar = np.dot(projection_matrix, M)
            ar = ar/ar[2]
            x, y = ar[0], ar[1]

            U = np.dot(A, ar)
            U = U/U[2]
            u, v = U[0], U[1]

            t = np.square(x) + np.square(y)
            u1 = u + (u-u0)*(k1*t + k2*(np.square(t)))
            v1 = v + (v-v0)*(k1*t + k2*(np.square(t)))

            reprojection_error[i] = point[0]-u1
            i = i + 1
            reprojection_error[i] = point[1]-v1
            i = i + 1

    return reprojection_error


def get_reprojection_error_Optimized(image_points, world_points, A, R, t, k1, k2):
    error = 0
    reprojected_points = []

    projection_matrix = np.zeros((3, 4))
    projection_matrix[:, :-1] = R
    projection_matrix[:, -1] = t

    #c1, c2 are the principal points
    c1, c2 = A[0, 2], A[1, 2]
    
    for point, world_point in zip(image_points, world_points):
        M = np.array([[world_point[0]], [world_point[1]], [0], [1]])
        #print('M', M)
        ar = np.dot(projection_matrix, M)
        ar = ar/ar[2]
        x, y = ar[0], ar[1]

        U = np.dot(A, ar)
        U = U/U[2]
        u, v = U[0], U[1]
        #from equations (11), (12) present in section 3.3 of paper
        t = np.square(x) + np.square(y)
        u1 = u + (u-c1)*(k1*t + k2*(t**2))
        v1 = v + (v-c2)*(k1*t + k2*(t**2))

        reprojected_points.append([u1, v1])

        error = error + np.sqrt((point[0]-u1)**2 + (point[1]-v1)**2)

    return error, reprojected_points


#def get_reprojection_error(image_points, world_points, A, R, t):
    # Project world points onto image plane
   # num_points = len(world_points)
   # projected_points = np.zeros((num_points, 2))
    #for i in range(num_points):
   #     projected_points[i] = np.dot(A, np.dot(R, world_points[i]) + t)

    # Calculate error
    #error = 0
   # for i in range(num_points):
    #    diff = projected_points[i] - image_points[i]
    #    error += np.linalg.norm(diff)
    #return error


def get_reprojection_error_(image_points, world_points, A, R, t):
    #world_points = np.resize(world_points, (3,))
    

    # Projecting world points onto image plane to find the error


    projection_matrix = np.zeros((3, 4))
    projection_matrix[:, :-1] = R
    projection_matrix[:, -1] = t

    X = np.dot(A, projection_matrix)
    error = 0
    for point, world_pt in zip(image_points, world_points):
        Projected_points = np.array([[world_pt[0]], [world_pt[1]], [0], [1]])
        image_point = np.array([[point[0]], [point[1]], [1]])
        projection_point = np.dot(X, Projected_points)
        projection_point = projection_point/projection_point[2]
        #calculate error
        difference = image_point - projection_point
        error += np.linalg.norm(difference)
        
    return error 
   


def Intrinsic_Parameters(b):

    """
    Calculates the intrinsic matrix from the b vector according to Appendix A
    in the reference paper.
    """
    
    #Matrix b is symmetric and defined by 6D vector
    #b11 = b[0]
    #b12 = b[1]
    #b22 = b[2]
    #b13 = b[3]
    #b23 = b[4]
    #b33 = b[5]
    
    v0_numerator = (b[0][1]*b[0][3] - b[0][0]*b[0][4])
    v0_denominator = (b[0][0]*b[0][2] - b[0][1]**2)

    v0 = v0_numerator/v0_denominator
    lmda = b[0][5] - (b[0][3]**2 +
                       v0*(b[0][1]*b[0][3] - b[0][0]*b[0][4]))/b[0][0]
    alpha = math.sqrt(lmda/b[0][0])
    beta = math.sqrt(lmda*b[0][0]/(b[0][0]*b[0][2] - b[0][1]**2))
    gamma = (-1*b[0][1]*alpha**2*beta)/(lmda)
    u = (gamma*v0)/beta - (b[0][3]*alpha**2)/lmda

    print("u - ", u)
    print("v - ", v0)
    print("lmda - ", lmda)
    print("alpha - ", alpha)
    print("beta - ", beta)
    print("gamma - ", gamma)

    #print(np.array([[alpha, gamma, u], 0, beta, v], [0, 0, 1]))

    A = convert_vector_to_matrix([alpha, gamma, beta, u, v0])
    return A, lmda



def b_matrix(V):
    #ESTIMATING b using Vb = 0 given in 3.1 part of paper
    U, s, Vh = np.linalg.svd(V, full_matrices=True, compute_uv=True, hermitian=False)
    
    #here we have value of b as 1x6 matrix
    b = Vh[-1:]
    print('b', b)
    return b



def get_V_matrix(H, V):
    V12 = [H[0][0]*H[0][1], (H[0][0]*H[1][1] + H[1][0]*H[0][1]), H[1][0]*H[1][1],
            (H[2][0]*H[0][1] + H[0][0]*H[2][1]), (H[2][0]*H[1][1] + H[1][0]*H[2][1]), H[2][0]*H[2][1]]
    
    #Here we calculate matrix V from 3.1 section of paper

    V1 = H[0][0]*H[0][0] - H[0][1]*H[0][1]
    V2 = (H[0][0]*H[1][0] - H[0][1]*H[1][1]) + (H[0][0]*H[1][0] - H[0][1]*H[1][1])
    V3 = H[1][0]*H[1][0] - H[1][1]*H[1][1]
    V4 = (H[2][0]*H[0][0] - H[0][1]*H[2][1]) + (H[2][0]*H[0][0] - H[0][1]*H[2][1])
    V5 = (H[2][0]*H[1][0] - H[1][1]*H[2][1]) + (H[2][0]*H[1][0] - H[1][1]*H[2][1])
    V6 = H[2][0]*H[2][0] - H[2][1]*H[2][1]

    v = np.hstack((V1, V2, V3, V4, V5, V6))

    V.append(V12)
    V.append(v)
    

    
def calculate_Homography(corners, world_points):
    n = corners.shape[0]
    src = np.asarray(world_points[: n])  
    dst = np.asarray(corners[: n])  

    #calculating L matrix which is a 2nx9 matrix.
    L = np.zeros((2*n, 9))

    i = 0
    for (src_point, dst_point) in zip(src, dst):

        L[i][0], L[i][1], L[i][2] = -src_point[0], -src_point[1], -1
        L[i+1][0], L[i+1][1], L[i+1][2] = 0, 0, 0

        L[i][3], L[i][4], L[i][5] = 0, 0, 0
        L[i+1][3], L[i+1][4], L[i+1][5] = -src_point[0], -src_point[1], -1

        L[i][6], L[i][7], L[i][8] = src_point[0]*dst_point[0], src_point[1]*dst_point[0], dst_point[0]
        L[i+1][6], L[i+1][7], L[i+1][8] = src_point[0]*dst_point[1], src_point[1]*dst_point[1], dst_point[1]

        i += 2

    U, S, V = np.linalg.svd(L, full_matrices=True)
    h = V[-1:]
    h.resize((3, 3))

    H = h/h[2, 2]

    return H


def main():
   
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--basepath', default="Data",
                        help='Path of chessboard images')

    Args = Parser.parse_args()
    dirpath = Args.basepath
    completepath = str(dirpath) + str("/*.jpg")
    images = sorted(glob(completepath))

    ap = argparse.ArgumentParser()
    ap.add_argument('--basepath', default="Output", help='Path of new chessboard images')
    Args = Parser.parse_args()
    dirpath = Args.basepath
    completepath = str(dirpath) + str("/*.jpg")
    Output_images = sorted(glob(completepath))

    V = []

    x, y = np.meshgrid(range(9), range(6))
    world_points = np.hstack((x.reshape(54, 1), y.reshape(
        54, 1))).astype(np.float32)
    world_points = world_points*21.5
    world_points = np.asarray(world_points)

    img_pts = []
    homography_matrices = []


    for imagepath in images:
        image = cv2.imread(imagepath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        
        
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret == True:
            corners = corners.reshape(-1, 2)
            cv2.drawChessboardCorners(gray, (9, 6), corners, ret)

            homography_matrix = calculate_Homography(corners, world_points)

            img_pts.append(corners)
            homography_matrices.append(homography_matrix)

            get_V_matrix(homography_matrix, V)

        
    
    V = np.asarray(V)
    b = b_matrix(V)

    K, lmda = Intrinsic_Parameters(b)
    print("\nInitial estimation of Calibration matrix is \n\n{}".format(K))

    error = 0
    for image_points, homography_matrix in zip(img_pts, homography_matrices):
        R, t = compute_extrinsic(K, homography_matrix)

        reprojection_error = get_reprojection_error_(image_points, world_points, K, R, t)

        error = error + reprojection_error
    #calculating the error
    error = error/(13*9*6)
    print("\nMean Reprojection error before optimization is \n{}".format(error))

    intrinsic = [K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[0, 1], 0, 0]
    
    res = opt.least_squares(fun=optimized_cost_function, x0=intrinsic,
                            method="lm", args=[img_pts, world_points, homography_matrices])

    

    K_after_optimization = np.zeros(shape=(3, 3))
    K_after_optimization[0, 0], K_after_optimization[1, 1], K_after_optimization[0, 2], K_after_optimization[1, 2], K_after_optimization[0, 1], K_after_optimization[2,
                                                                           2] = res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], 1

    k1_after_optimization, k2_after_optimization = res.x[5], res.x[6]

    print("\nCalibration matrix after optimization is \n\n{}".format(K_after_optimization))
    print("\nDistortion coefficients after optimization: \n{}, {}".format(k1_after_optimization, k2_after_optimization))

    error_after_optimization = 0
    calibration_points = []
    for image_points, homography_matrix in zip(img_pts, homography_matrices):
        R, t = compute_extrinsic(K_after_optimization, homography_matrix)

        reprojection_error, reprojected_points = get_reprojection_error_Optimized(
            image_points, world_points, K_after_optimization, R, t, k1_after_optimization, k2_after_optimization)
        calibration_points.append(reprojected_points)

        error_after_optimization = error_after_optimization + reprojection_error

    error_after_optimization = error_after_optimization/(702)
    print("Mean Reprojection error after optimization is ", error_after_optimization[0])

    projection_from_3D_to_2D(img_pts, calibration_points, images)


    SourceRefImagePath = 'Output'
    dist = np.array(
        [2.90493410e-01, -2.42737867e+00,  2.70529994e-03, 9.61801276e-04,  6.52472281e+00])
    images = os.listdir(SourceRefImagePath)
    for img in images:
        source_image_path = SourceRefImagePath + "/" + img
        fetched_image = cv2.imread(source_image_path)
        h, w = fetched_image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist,
                                                          (w, h), 1,
                                                          (w, h))
        dst = cv2.undistort(fetched_image, K, dist, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        cv2.imwrite("Undistort/{}".format(img), dst)

if __name__ == '__main__':
    main()
