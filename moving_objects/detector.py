###########################################################################
###### EECS106B final project Module 1: moving objects detection ##########
###########################################################################

# func name using underscore, variable name using camelCase
# img1 is train, img2 is query
import cv2
import numpy as np
# import matplotlib.pyplot as plt
import skvideo.io as io

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

    def match(self):
        """
        Extract the features of self.img1 and self.img2 using FAST detector and BRIEF descriptor
        and match features using KNN and ratio test
        :return: keypoints in img1, keypoints in img2, and good matches
        """
        kp1 = self.fDetector.detect(self.img1, None)
        kp1, des1 = self.fDescriptor.compute(self.img1, kp1)

        kp2 = self.fDetector.detect(self.img2, None)
        kp2, des2 = self.fDescriptor.compute(self.img2, kp2)

        matches = self.fMatcher.knnMatch(des2, des1, k=2)

        # apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        return kp1, kp2, good

    def compute3D(self, matches, kp1, kp2):
        """
        compute the local 3d coordinates of keypoints
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

    def img_to_3D(self, low=100, high=6000):
        """
        compute the local 3d coordinates of self.img1, ignore depth < low or > high
        :param low: the lower bound, depth values below low is ignored
        :param high: the upper bound, depth values above high is ignored
        :return: matrix containing the local 3D coordinates of the scene
        """
        h, w = self.img1.shape[:2]
        pc1 = np.zeros((h*w, 3))
        cmInv = np.linalg.pinv(self.camMat)
        cm = cmInv.T

        p = np.ones((w, 3))
        p[:, 0] = np.arange(w)

        for i in range(h):
            p[:, 1] = np.ones(w) * i
            pcm = p @ cm

            d1 = self.depth1[i]
            d1 = np.where(np.logical_and(d1 > low, d1 < high), d1, 0)
            d1 = np.diag(d1.flatten())

            pc1[i * w : (i + 1) * w] = d1 @ pcm

        return pc1

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


    def detect_2D(self, kp1, kp2, good):
        """
        detect the moving objects based on img1 and img2 using homography
        assuming that the camera motion is small or no motion, otherwise, the camera motion will be counted
        :param kp1: keypoints in img1
        :param kp2: keypoints in img2
        :param good: good matches from ratio test
        :return: an image of moving objects
        """
        # extract pixel coordinates of good matched features
        num = len(good)
        pts1 = np.zeros((num, 2), dtype=np.float32)
        pts2 = np.zeros((num, 2), dtype=np.float32)

        for i, match in enumerate(good):
            pts1[i, :] = kp1[match.trainIdx].pt
            pts2[i, :] = kp2[match.queryIdx].pt

        # compute homography
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)
        size = self.img1.shape[:2]
        img1 = cv2.warpPerspective(self.img1, H, (size[1], size[0]))

        # compare the warpped image with self.img2
        # apply gaussianblur to smooth the images
        img1Blur = cv2.GaussianBlur(img1, (5, 5), 0)
        img2Blur = cv2.GaussianBlur(self.img2, (5, 5), 0)

        # bias/gain normalization
        img1 = (img1Blur - np.mean(img1Blur)) / np.std(img1Blur)
        img2 = (img2Blur - np.mean(img2Blur)) / np.std(img2Blur)

        diff = (np.abs(img1 - img2) * 255).astype(np.uint8)
        if np.max(diff) - np.min(diff) >= 250:
            result = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        else:
            result = np.zeros(img2.shape)

        return result

    def detect_3D(self, kp1, kp2, good):
        """
        detect the moving objects based on depth1 and depth2 using multiview geometry
        assuming that the camera is calibrated
        :param kp1: keypoints in img1
        :param kp2: keypoints in img2
        :param good: good matches from ratio test
        :return: an image of moving objects
        """

        # estimate rigid body transformation
        kp1, kp2 = self.compute3D(good, kp1, kp2)
        R, T, _ = self.estimate_rigid_transform(kp1, kp2)

        # create a point cloud, dim = (h*w, 3)
        low, high = 0, np.max(self.depth1)
        pc1 = self.img_to_3D(low, high)

        # project point cloud one onto the projection plane for frame two
        pc1 = self.camMat @ (R @ pc1.T + T)

        # convert the point cloud in the local coordinate frame of img2 into a depth array
        h, w = self.depth1.shape[:2]
        z = (pc1[2]).astype(np.uint16)
        coords = (pc1[:2] / pc1[2]).astype(int)

        depth = np.zeros((h, w))
        for i in range(pc1.shape[1]):
            c, r = coords[:, i]
            if 0 <= c < w and 0 <= r < h:
                depth[r, c] = z[i]

        # smooth the depth array, apply bias/gain normalization
        d1Blur = cv2.GaussianBlur(depth, (9, 7), 0)
        d1 = (d1Blur - np.mean(d1Blur)) / np.std(d1Blur)
        d2 = (self.depth2 - np.mean(self.depth2)) / np.std(self.depth2)

        # compare the reprojected depth array to the original depth two
        diff = np.abs(d1 - d2)
        diff = (diff * 255).astype(np.uint8)
        return cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def detect(self):
        """
        detect the moving objects based on depth1 and depth2 using multiview geometry, img1 and img2 using homography
        assuming that the camera is calibrated, and the camera motion is small
        :return: an image of moving objects
        """
        kp1, kp2, matches = self.match()
        res2D = self.detect_2D(kp1, kp2, matches)
        res3D = self.detect_3D(kp1, kp2, matches)

        # apply l1 norm
        if np.array_equal(res2D, np.zeros(res2D.shape)):
            return res2D
        return ((res2D * 0.5 + res3D * 0.5) > 200) * 255

def main():
    filename = 'videos/walking.MOV'
    video = io.vread(filename, as_grey=True)

    depths = []
    for i in range(70, 340):
        filename = 'imgs/{}_cam-depth_array_.png'.format(i)
        d = cv2.imread(filename, -1)
        d = cv2.GaussianBlur(d, (9, 7), 0)
        depths.append(d)

    imgs = []
    offset = 5
    failure = 0
    for i, frame in enumerate(video):
        print(i)
        frame = frame[:, :, 0]
        if i < offset:
            res = np.zeros(frame.shape)
            res2 = np.zeros(frame.shape)
            res3 = np.zeros(frame.shape)
        else:
            try:
                detector = Detector(video[i - offset], frame, depths[i - offset], depths[i])
                res = detector.detect()
                kp1, kp2, matches = detector.match()
                res2 = detector.detect_2D(kp1, kp2, matches)
                res3 = detector.detect_3D(kp1, kp2, matches)
            except:
                failure += 1
                res = np.zeros(frame.shape)
                res2 = np.zeros(frame.shape)
                res3 = np.zeros(frame.shape)

        top = np.hstack((frame, res))
        bottom = np.hstack((res2, res3))
        img = np.vstack((top, bottom))
        imgs.append(img)
    print("failure: ", failure)
    io.vwrite('walking_detected.MOV', imgs)

if __name__ == "__main__":
    main()