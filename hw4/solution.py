import numpy as np
import cv2
import math
import random

# ratio_thres=0.6, orient_agreement=30, scale_agreement=0.5
def RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement):
    """
    This function takes in `matched_pairs`, a list of matches in indices
    and return a subset of the pairs using RANSAC.
    Inputs:
        matched_pairs: a list of tuples [(i, j)],
            indicating keypoints1[i] is matched
            with keypoints2[j]
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        *_agreement: thresholds for defining inliers, floats
    Output:
        largest_set: the largest consensus set in [(i, j)] format

    HINTS: the "*_agreement" definitions are well-explained
           in the assignment instructions.
    """
    assert isinstance(matched_pairs, list)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    ## START

    largest_set = []
    for i in range(10):  # 10번 반복
        rand = random.randrange(0, len(matched_pairs)) # 랜덤한 하나의 숫자를 선택
        choice = matched_pairs[rand] 
        choice_point1, choice_point2 = choice[0], choice[1] # matching되는 point1, point2 
        orientation = (keypoints1[choice_point1][3] - keypoints2[choice_point2][3]) % (2 * math.pi)  # 방향 계산
        scale = keypoints2[choice_point2][2] / keypoints1[choice_point1][2]  # ratio 계산
        temp = []
        for j in range(len(matched_pairs)):  # 맨 처음 뽑은 점을 제외한 모든 점들에 대해 계산하기
            if j is not rand:
                temp_orientation = abs((keypoints1[matched_pairs[j][0]][3] - keypoints2[matched_pairs[j][1]][3]) % (2 * math.pi))
                temp_scale = abs(keypoints2[matched_pairs[j][1]][2] /  keypoints1[matched_pairs[j][0]][2])
                # 방향과 scale 체크하기
                if (orientation - orient_agreement / 6) < temp_orientation < (orientation + orient_agreement / 6):
                    if scale - scale * scale_agreement < temp_scale < scale + scale * scale_agreement:
                        temp.append([i, j])
        # 만약 더 크다면, 갱신
        if len(temp) > len(largest_set):
            largest_set = temp
    # 좌표쌍으로 바꿔주기
    for i in range(len(largest_set)):
        largest_set[i] = (matched_pairs[largest_set[i][1]][0],
                          matched_pairs[largest_set[i][1]][1])
    ## END
    assert isinstance(largest_set, list)
    return largest_set


def FindBestMatches(descriptors1, descriptors2, threshold):
    """
    This function takes in descriptors of image 1 and image 2,
    and find matches between them. See assignment instructions for details.
    Inputs:
        descriptors: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
    Outputs:
        matched_pairs: a list in the form [(i, j)] where i and j means
                       descriptors1[i] is matched with descriptors2[j].
    """
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    # START
    # the following is just a placeholder to show you the output format

    #i 와 j에 대한 각도 계산
    matched_pairs = []
    for i in range(len(descriptors1)):
        angles = []
        for j in range(len(descriptors2)):
            cos_angle = np.dot(descriptors1[i], descriptors2[j])
            angle = np.arccos(cos_angle)
            angles.append((angle, j))  # 각도와 인덱스를 함께 저장

        angles.sort() # 각도 기준으로 정렬
        
        # 최소 각도와 두 번째 최소 각도의 비율 검사
        if (angles[0][0] / angles[1][0]) <= threshold:
            matched_pairs.append((i, angles[0][1]))  
 
    return matched_pairs


def KeypointProjection(xy_points, h):
    """
    This function projects a list of points in the source image to the
    reference image using a homography matrix `h`.
    Inputs:
        xy_points: numpy array, (num_points, 2)
        h: numpy array, (3, 3), the homography matrix
    Output:
        xy_points_out: numpy array, (num_points, 2), input points in
        the reference frame.
    """
    assert isinstance(xy_points, np.ndarray)
    assert isinstance(h, np.ndarray)
    assert xy_points.shape[1] == 2
    assert h.shape == (3, 3)

    # START
    hc_xys = np.pad(xy_points, pad_width=((0,0), (0,1)) , mode = 'constant' , constant_values=1) #np.pad를 사용해서 nx2 행렬을 nx3으로 변환함. 이는 z=1 homogeneous로 변환.
    xys_p = h @ hc_xys.T # matrix 곱셈 -> 변환된 (x,y,z) matrix임
    z_cor = np.where(xys_p[-1,:] ==0, 0.000001, xys_p[-1:]) #곱한 결과에서 z=0인 곳을 0.000001로 변환함 
    hc_xys_p = xys_p/z_cor # z=1로 정규화함
    xys_p=hc_xys_p[:-1, :] # (x,y)쌍만 저장
    
    #print(xys_p.T)
    return xys_p.T # transpose후 return -> point x 2 shape
    # END

def RANSACHomography(xy_src, xy_ref, num_iter, tol):
    max_inliers_count = 0
    best_h = None
    N = xy_src.shape[0]

    for i in range(num_iter):
        sample_indices = np.random.choice(N, 4, replace=False) # 랜덤한 4개의 점을 뽑기
        src_sample = xy_src[sample_indices]
        ref_sample = xy_ref[sample_indices]
        
        x1,y1 = src_sample[0]
        x2,y2 = src_sample[1]
        x3,y3 = src_sample[2]
        x4,y4 = src_sample[3]
        
        x_1, y_1 = ref_sample[0]
        x_2, y_2 = ref_sample[1]
        x_3, y_3 = ref_sample[2]
        x_4, y_4 = ref_sample[3]
        
        A = np.array([
                    [x1 , y1 , 1 , 0 , 0 , 0 , -1*(x_1)*x1 , -1*(x_1)*y1 , -1*(x_1)],
                    [0 , 0 , 0 , x1 , y1 , 1 , -1*(y_1)*x1 , -1*(y_1)*y1 , -1*(y_1)],
                    [x2 , y2 , 1 , 0 , 0 , 0 , -1*(x_2)*x2 , -1*(x_2)*y2 , -1*(x_2)],
                    [0 , 0 , 0 , x2 , y2 , 1 , -1*(y_2)*x2 , -1*(y_2)*y2 , -1*(y_2)],
                    [x3 , y3 , 1 , 0 , 0 , 0 , -1*(x_3)*x3 , -1*(x_3)*y3 , -1*(x_3)],
                    [0 , 0 , 0 , x3 , y3 , 1 , -1*(y_3)*x3 , -1*(y_3)*y3 , -1*(y_3)],
                    [x4 , y4 , 1 , 0 , 0 , 0 , -1*(x_4)*x4 , -1*(x_4)*y4 , -1*(x_4)],
                    [0 , 0 , 0 , x4 , y4 , 1 , -1*(y_4)*x4 , -1*(y_4)*y4 , -1*(y_4)]]) #8 * 9 A matrix
        
        ATA = A.T @ A # matrix 곱
        eigenvalues, eigenvectors = np.linalg.eig(ATA) # 고유벡터, 고윳값 구하기
        H = eigenvectors[:, np.argmin(eigenvalues)] # 제일 작은 eigenvalue에 mapping되는 eigenvector 구하기
        H /= H[8] # 정규화
        H = H.reshape(3,3) # 3x3 matrix로 만들기

        xy_proj = KeypointProjection(xy_src, H) # projection
        errors = np.sqrt((xy_proj - xy_ref)**2).sum(axis=1)
        inliers = errors <= tol 
        inlier_count = np.sum(inliers) # tol보다 작은 값들의 수 찾기
        
        if inlier_count > max_inliers_count: # homography 갱신
            best_h = H
            max_inliers_count = inlier_count
            
    
    assert isinstance(best_h, np.ndarray)
    assert best_h.shape == (3, 3)
    return best_h



def FindBestMatchesRANSAC(
        keypoints1, keypoints2,
        descriptors1, descriptors2, threshold,
        orient_agreement, scale_agreement):
    """
    Note: you do not need to change this function.
    However, we recommend you to study this function carefully
    to understand how each component interacts with each other.

    This function find the best matches between two images using RANSAC.
    Inputs:
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        descriptors1, 2: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
        orient_agreement: in degrees, say 30 degrees.
        scale_agreement: in floating points, say 0.5
    Outputs:
        matched_pairs_ransac: a list in the form [(i, j)] where i and j means
        descriptors1[i] is matched with descriptors2[j].
    Detailed instructions are on the assignment website
    """
    orient_agreement = float(orient_agreement)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    matched_pairs = FindBestMatches(
        descriptors1, descriptors2, threshold)
    matched_pairs_ransac = RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement)
    return matched_pairs_ransac
