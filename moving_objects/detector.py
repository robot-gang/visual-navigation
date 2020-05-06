###########################################################################
###### EECS106B final project Module 1: moving objects detection ##########
###########################################################################

# func name using underscore, variable name using camelCase
# img1 is train, img2 is query
import cv2
import numpy as np
import matplotlib.pyplot as plt

detectors = {'fast' : cv2.FastFeatureDetector_create(),
             'star' : cv2.xfeatures2d.StarDetector_create(),
             'brisk': cv2.BRISK_create(),
             'orb'  : cv2.ORB_create(),
             'mser' : cv2.MSER_create(),
             'gftt' : cv2.GFTTDetector_create(),
             'blob' : cv2.SimpleBlobDetector_create()}

descriptors = {'brief' : cv2.xfeatures2d.BriefDescriptorExtractor_create(),
               'orb'   : cv2.ORB_create(),
               'daisy' : cv2.xfeatures2d.DAISY_create(),
               'boost' : cv2.xfeatures2d.BoostDesc_create(),
               'freak' : cv2.xfeatures2d.FREAK_create(),
               'latch' : cv2.xfeatures2d.LATCH_create(),
               'lucid' : cv2.xfeatures2d.LUCID_create(),
               'vgg'   : cv2.xfeatures2d.VGG_create()}

matchers = {'bruteForce': cv2.BFMatcher(cv2.NORM_HAMMING),
            'flann'     : cv2.FlannBasedMatcher_create()}
cameraMat = np.array([[317.73273, 0, 319.9013], [0, 317.73273, 177.84988], [0, 0, 1]])

class Detector:
    def __init__(self, im1, im2, depth1=None, depth2=None, cameraMatrix=cameraMat, detector='fast', descriptor='brief', matcher='bruteForce'):
        self.img1 = im1
        self.img2 = im2
        self.camMat = cameraMatrix
        self.depth1 = depth1
        self.depth2 = depth2

        self.fDetector = detectors[detector]
        self.fDescriptor = descriptors[descriptor]
        self.fMatcher = matchers[matcher]

    def feature_extraction(self):
        """
        Extract the features of self.img1 and self.img2 using FAST detector and BRIEF descriptor
        :return:
        """
        kp1 = self.fDetector.detect(self.img1, None)
        kp1, des1 = self.fDescriptor.compute(self.img1, kp1)

        kp2 = self.fDetector.detect(self.img2, None)
        kp2, des2 = self.fDescriptor.compute(self.img2, kp2)

        return kp1, des1, kp2, des2

    def compute3D(self, matches, kp1, kp2):
        """
        compute the local 3d points
        :param matches: the feature matched using BFmatching
        :param kp1: the keypoints in self.img1
        :param kp2: the keypoints in self.img2
        :return: ndarray of points in 3D, dim = n X 3
        """
        # find the pixel coordinates of keypoints, dim n x 3
        kp1Pix = [kp1[m.trainIdx].pt for m in matches]
        kp1Pix = np.hstack((kp1Pix, np.ones((len(kp1Pix), 1))))

        kp2Pix = [kp2[m.queryIdx].pt for m in matches]
        kp2Pix = np.hstack((kp2Pix, np.ones((len(kp2Pix), 1))))

        # get the depth info of each point, n x n
        xIdx1, yIdx1 = (kp1Pix[:, 0]).astype(int), (kp1Pix[:, 1]).astype(int)
        d1 = np.diag(self.depth1[yIdx1, xIdx1])
        xIdx2, yIdx2 = (kp2Pix[:, 0]).astype(int), (kp2Pix[:, 1]).astype(int)
        d2 = np.diag(self.depth2[yIdx2, xIdx2])

        cmInv = np.linalg.pinv(self.camMat)
        kp1Local3D = d1 @ kp1Pix @ cmInv.T
        kp2Local3D = d2 @ kp2Pix @ cmInv.T

        return kp1Local3D, kp2Local3D

    def get_rigid_transform(self, A, B):
        """
        Find rigid body transformation between A and B
        :param A: 3 X n matrix of points
        :param B: 3 x n matrix of points
        :return: rigid body transformation between A and B
        code is from https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
        solve the problem RA + t = B
        """

        assert len(A) == len(B)
        num_rows, num_cols = A.shape
        if num_rows != 3:
            raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))
        [num_rows, num_cols] = B.shape
        if num_rows != 3:
            raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

        # find mean column wise
        centroid_A = np.mean(A, axis=1).reshape((3, 1))
        centroid_B = np.mean(B, axis=1).reshape((3, 1))

        # subtract mean
        Am = A - centroid_A
        Bm = B - centroid_B

        # compute covariance matrix
        H = Am @ np.transpose(Bm)

        # find rotation
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # special reflection case, when the determinant is -1
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        t = -R @ centroid_A + centroid_B

        return R, t.reshape((3, 1))


    def estimate_rigid_transform(self, kp1, kp2, numIter = 100, tolerance = 50, inlierBound = 10):
        """
        find the best R, T between kp2 and kp1, R @ kp1 + t = kp2
        :param kp1: n X 3 matrix of 3d points in img1
        :param kp2: n X 3 matrix of 3d points in img2
        :param numIter: max number of iterations
        :param tolerance: bound used to check the quality of transformation, DEFAULT = 3cm
        :param inlier_bound: the minimum number of inliers required
        :return: best rigid transformation R, T, index of inliers
        """

        assert len(kp2) == len(kp1)

        accuracy = 0
        bestInliers = None
        for i in range(numIter):
            # create 3 random non-repeated indices of the correspondence points
            idx = np.random.choice(len(kp2), 3, replace=False)
            A = kp1[idx].T
            B = kp2[idx].T
            R, t = self.get_rigid_transform(A, B)

            error = np.linalg.norm(R @ kp1.T + t - kp2.T, axis = 0)
            inliers = error < tolerance
            acc = np.sum(inliers)

            if acc > accuracy and acc >= inlierBound:
                accuracy = acc
                bestInliers = inliers

        if bestInliers is None:
            return None, None, None
        else:
            idx = np.where(bestInliers > 0)[0]
            R, T = self.get_rigid_transform(kp1.T[:, idx], kp2.T[:, idx])
            return R, T, bestInliers


    def create_mask(self, numFeatures = 400, minFeatures = 10):
        """
        create a mask for the moving objects
        :param numFeatures: the number of features used in calculation ranked distance
        :param minFeatures: minimum number of features before stopping RANSAC
        :return: a mask
        """
        # extract and match features
        kp1, des1, kp2, des2 = self.feature_extraction()
        # matches = self.fMatcher.match(des2, des1)
        matches = self.fMatcher.knnMatch(des2, des1, k=3)
        matches = sorted(matches, key=lambda x: x.distance)[:numFeatures]

        if self.depth1 is not None:
            # compute local 3D coordinates of features
            kp1, kp2 = self.compute3D(matches, kp1, kp2)

        Rs, Ts,  = [], []

        while True:
            R, T, inliers = self.estimate_rigid_transform(kp1, kp2)
            fig, ax = plt.subplot(1, 2, figsize=(10, 5))
            ax[0].imshow(self.img1)
            ax[0].scatter(kp1[inliers][:, 0], kp1[inliers][:, 1])

            ax[1].imshow(self.img2)
            ax[1].scatter(kp2[inliers][:, 0], kp2[inliers][:, 1])
            plt.imshow()


            Rs.append(R)
            Ts.append(T)

            idx = np.where(inliers==0)
            kp1 = kp1[idx]
            kp2 = kp2[idx]

            if R is None and len(idx[0]) < minFeatures:
                break

