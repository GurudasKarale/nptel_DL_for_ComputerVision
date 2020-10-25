import numpy as np

class Affine():

    def create_test_case(self, outlier_rate=0.9):

        # Randomly generate affine transformation
        # A is a 2x2 matrix, the range of each value is from -2 to 2
        A = 4 * np.random.rand(2, 2) - 2

        # % t is a 2x1 VECTOR, the range of each value is from -10 to 10
        t = 20 * np.random.rand(2, 1) - 10

        # Set the number of points in test case
        num = 1000

        # Compute the number of outliers and inliers respectively
        outliers = int(np.round(num * outlier_rate))
        inliers = int(num - outliers)

        # Gernerate source points whose scope from (0,0) to (100, 100)
        pts_s = 100 * np.random.rand(2, num)
        # Initialize warped points matrix
        pts_t = np.zeros((2, num))

        # Compute inliers in warped points matrix by applying A and t
        pts_t[:, :inliers] = np.dot(A, pts_s[:, :inliers]) + t

        # Generate outliers in warped points matrix
        pts_t[:, inliers:] = 100 * np.random.rand(2, outliers)

        # Reset the order of warped points matrix,
        # outliers and inliers will scatter randomly in test case
        rnd_idx = np.random.permutation(num)
        pts_s = pts_s[:, rnd_idx]
        pts_t = pts_t[:, rnd_idx]

        return A, t, pts_s, pts_t

    def estimate_affine(self, pts_s, pts_t):

        # Get the number of corresponding points
        pts_num = pts_s.shape[1]

        # Initialize the matrix M,
        # M has 6 columns, since the affine transformation
        # has 6 parameters in this case
        M = np.zeros((2 * pts_num, 6))

        for i in range(pts_num):
            # Form the matrix m
            temp = [[pts_s[0, i], pts_s[1, i], 0, 0, 1, 0],
                    [0, 0, pts_s[0, i], pts_s[1, i], 0, 1]]
            M[2 * i: 2 * i + 2, :] = np.array(temp)

        # Form the matrix b,
        # b contains all known target points
        b = pts_t.T.reshape((2 * pts_num, 1))

        try:
            # Solve the linear equation
            theta = np.linalg.lstsq(M, b,rcond=None)[0]

            # Form the affine transformation
            A = theta[:4].reshape((2, 2))
            t = theta[4:]
        except np.linalg.linalg.LinAlgError:
            # If M is singular matrix, return None
            # print("Singular matrix.")
            A = None
            t = None

        return A, t


# The number of iterations in RANSAC
ITER_NUM = 2000


af=Affine()
A_true, t_true, pts_s, pts_t = af.create_test_case()

#########Ransac
class Ransac():

    def __init__(self, K=3, threshold=1):

        self.K = K
        self.threshold = threshold

    def residual_lengths(self, A, t, pts_s, pts_t):

        if not(A is None) and not(t is None):
            # Calculate estimated points:
            # pts_esti = A * pts_s + t
            pts_e = np.dot(A, pts_s) + t

            # Calculate the residual length between estimated points
            # and target points
            diff_square = np.power(pts_e - pts_t, 2)
            residual = np.sqrt(np.sum(diff_square, axis=0))
        else:
            residual = None

        return residual

    def ransac_fit(self, pts_s, pts_t):

        # Create a Affine instance to do estimation
        af = Affine()

        # Initialize the number of inliers
        inliers_num = 0

        # Initialize the affine transformation A and t,
        # and a vector that stores indices of inliers
        A = None
        t = None
        inliers = np.array([], dtype=int)

        min_residual = None

        for i in range(ITER_NUM):
            # Randomly generate indices of points correspondences
            idx = np.random.randint(0, pts_s.shape[1], (self.K, 1))
            if len(np.unique(idx)) < self.K:
                continue

            # Estimate affine transformation by these points
            A_tmp, t_tmp = af.estimate_affine(pts_s[:, idx], pts_t[:, idx])

            # Calculate the residual by applying estimated transformation
            residual = self.residual_lengths(A_tmp, t_tmp, pts_s, pts_t)


            if not(residual is None):
                inliers_tmp = np.where(residual < self.threshold)
                inliers_num_tmp = len(inliers_tmp[0])

                if inliers_num_tmp > self.K and inliers_num_tmp > inliers_num:
                    inliers_num = inliers_num_tmp
                    inliers = inliers_tmp
                    A = A_tmp
                    t = t_tmp
            else:
                pass

        return A, t, inliers

r=Ransac()
r.ransac_fit(pts_s,pts_t)
_,_,inn=r.ransac_fit(pts_s,pts_t)
v=list(inn)
vv=np.asarray(v)
print(v)
