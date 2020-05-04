###########################################################################
###### EECS106B final project Module 1: moving objects detection ##########
###########################################################################

# func name using underscore, variable name using camelCase
# img1 is train, img2 is query
import cv2
import numpy as np

detectors = {'fast' : cv2.FastFeatureDetector_create()}
descriptors = {'brief' : cv2.xfeatures2d.BriefDescriptorExtractor_create()}
matchers = {'bruteForce': cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)}

class Detector:
    def __init__(self, im1, im2, cameraMatrix, depth1=None, depth2=None, detector='fast', descriptor='brief', matcher='bruteForce'):
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
        kp1Pix = np.hstack((kp1Pix, np.ones(len(kp1Pix))))

        kp2Pix = [kp2[m.queryIdx].pt for m in matches]
        kp2Pix = np.hstack((kp2Pix, np.ones(len(kp2Pix))))

        # cur_2d = np.asarray([[kp1[match.queryIdx].pt[0], kp1[match.queryIdx].pt[1], 1] for match in matches]).T
        # prev_2d =  np.asarray([[kp2[match.trainIdx].pt[0], kp2[match.trainIdx].pt[1], 1] for match in matches]).T

        # get the depth info of each point, n x n
        xIdx1, yIdx1 = kp1Pix[:, 0], kp1Pix[:, 1]
        d1 = np.diag(self.depth1[yIdx1, xIdx1])
        xIdx2, yIdx2 = kp2Pix[:, 0], kp2Pix[:, 1]
        d2 = np.diag(self.depth2[yIdx2, xIdx2])

        # current_depth = np.diag(cur_depth[cur_2d[1].astype(int), cur_2d[0].astype(int)])
        # previous_depth = np.diag(self.prev_depth[prev_2d[1].astype(int), prev_2d[0].astype(int)])
        cmInv = np.linalg.pinv(self.camMat)
        kp1Local3D = d1 @ kp1Pix @ cmInv
        kp2Local3D = d2 @ kp2Pix @ cmInv
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
        centroid_A = np.mean(A, axis=1)
        centroid_B = np.mean(B, axis=1)

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


    def estimate_rigid_transform(self, kp1, kp2, numIter = 100, tolerance = 30, inlierBound = 10):
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
            return self.get_rigid_transform(kp1.T[:, idx], kp2.T[:, idx]), bestInliers


    def create_mask(self, numFeatures = 400, minFeatures = 10):
        """
        create a mask for the moving objects
        :param numFeatures: the number of features used in calculation ranked distance
        :param minFeatures: minimum number of features before stopping RANSAC
        :return: a mask
        """
        # extract and match features
        kp1, des1, kp2, des2 = self.feature_extraction()
        matches = self.fMatcher.match(des2, des1)
        matches = sorted(matches, key=lambda x: x.distance)[:numFeatures]

        if self.depth1 is not None:
            # compute local 3D coordinates of features
            kp1, kp2 = self.compute3D(matches, kp1, kp2)

        while True:
            R, t, inliers = self.estimate_rigid_transform(kp1, kp2)

            idx = np.where(inliers==0)
            kp1 = kp1[idx]
            kp2 = kp2[idx]

            if R is None and len(idx[0]) < minFeatures:
                break

