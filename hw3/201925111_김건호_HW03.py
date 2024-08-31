from PIL import Image
import math
import numpy as np

def gauss1d(sigma):
    #sigma에 6배를 한 후, 반올림합니다.
    round_up_sigma = round(sigma*6)
    
    #만약 짝수라면 1을 더해 홀수로 만듭니다.
    #round_up_sigma = round_up_sigma if round_up_sigma%2 else round_up_sigma+1 
    if round_up_sigma%2==0 :
        round_up_sigma+=1
        
    #map + lambda function과 list comprehension을 사용해서 1차원 배열을 만듭니다.
    arr=np.array([i for i in range(-(round_up_sigma//2), 1+ (round_up_sigma//2))])
    array_1d = np.array(list(map(lambda x: 1/math.sqrt(2*math.pi*sigma**2) * math.exp(-(x**2)/(2*sigma**2)), arr)))
    
    #합이 1이 되도록 정규화합니다.
    normalization_array_1d=array_1d/sum(array_1d)
    return normalization_array_1d

def gauss2d(sigma) : 
    # sigma값을 입력으로, 1차원 gaussian filter를 만들고, np.outer과 np.transpose를 통해 2차원 gaussian filter를 만듭니다.
    # np.outer후 결과값의 합은 어차피 1일 것이므로, 따로 정규화해줄 필요는 없습니다.
    # 왜냐하면, 모든 원소의 합이 1인 이미 정규화된 1차원 gaussain filer에, transpose후 outer product한 
    # 2차원 gaussian filer의 모든 원소의 합은 1일 것이기 때문입니다.
    array_1d=gauss1d(sigma)
    return np.outer(array_1d, np.transpose(array_1d)) 

def convolve2d(array, filter) :
    #input 변수의 dtype을 np.float32로 변경합니다.
    array=array.astype(np.float32)
    filter=filter.astype(np.float32)
    
    #convoultion_array를 list comprehension으로 만듭니다.
    height=len(array)
    width=len(array[0])
    convolution_array = np.array([[0 for _ in range(width)] for _ in range(height)] ,dtype=np.float32)
    
    # padding size를 계산 후, array에 padding을 추가합니다.
    padding_size=int((len(filter[0])-1)/2)
    padding_array = np.pad(array, ((padding_size, padding_size), (padding_size, padding_size)), 'constant', constant_values=0)
    
    # convolution 계산의 편의성을 위해 filter를 flip을 합니다.
    filter_size=len(filter)
    flipped_filter=np.flip(filter)
    
    # pad가 추가된 2차원 image에서 filter_size만큼 crop후, flip된 filter로 cross-correlation을 진행합니다.
    # filter를 flip후, cross-correlation한 결과는 convolution한 결과와 동일합니다.
    for row in range(height) : 
        for col in range(width) :
            crop_image=padding_array[row:row+filter_size, col : col+filter_size]
            convolution_array[row][col]=np.sum(crop_image * flipped_filter)
            
    return convolution_array

def gaussconvolve2d(array,sigma) :
    gaussian_filter=gauss2d(sigma)
    return convolve2d(array, gaussian_filter)

""" Return the gray scale gaussian filtered image with sigma=1.6
    Args:
        img: RGB image. Numpy array of shape (H, W, 3).
    Returns:
        res: gray scale gaussian filtered image (H, W).
"""
def reduce_noise(img):
    image1=img.convert('L') #gray scale로 image convert
    image1=np.asarray(image1) #np.array로 변환
    res=gaussconvolve2d(image1,1.6) #gaussian convolution - > gaussconvolve2d function 내부에서 np.float type으로 변경해줍니다.
    return res #res는 np.array, dtype=float32입니다.

def sobel_filters(img):
    """ Returns gradient magnitude and direction of input img.
    Args:
        img: Grayscale image. Numpy array of shape (H, W).
    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction of gradient at each pixel in img.
            Numpy array of shape (H, W).
    Hints:
        - Use np.hypot and np.arctan2 to calculate square root and arctan
    """

    sx = np.array([[1,0,-1],[2,0,-2], [1,0,-1]]) # sobel x filter 선언
    sy = np.array([[-1,-2,-1],[0,0,0], [1,2,1]]) # sobel y filter 선언
    
    gradient_x=convolve2d(img ,sx) # convolved2d function에서 pad 추가 및 flip도 해줍니다.
    gradient_y=convolve2d(img ,sy) # sobel filter와 img를 convolution해서 gradient_x, gradient_y를 얻습니다.
    G=np.hypot(gradient_x, gradient_y) # np.hypot을 활용하여, gradient 크기를 G에 저장합니다.
    G = G / G.max() * 255 # G 내부 값들은 255보다 클 수 있기에, 정규화합니다.
    theta=np.arctan2(gradient_y,gradient_x) # np.arctan2를 활용하여, gradient 방향에 대한 각을 theta에 저장합니다.
    print(theta)
    return (G, theta)

def non_max_suppression(G, theta):
    """ Performs non-maximum suppression.
    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).
    Returns:
        res: non-maxima suppressed image.
    """
    res=theta.copy() 
    res.fill(0) # 0으로 채운 새로운 배열을 선언합니다.

    height=len(theta)
    width=len(theta[0])
    for row in range(1,height-1) : 
        for col in range(1,width-1) : #테두리 부분은 검사하지 않습니다. -> 과제 설명 영상 기준 
            angle=theta[row,col] 
            if (angle>=np.pi/8 and angle<(np.pi*3)/8) or  (angle<-5*(np.pi/8) and angle>=-7*((np.pi*3)/8)): # 2시와 8시 방향을 봅니다.
                if G[row,col]>max(G[row+1,col-1], G[row-1,col+1]) : # 이웃들보다 크다면,
                    res[row,col]=G[row,col] # 값을 선택합니다.
            elif angle>=(np.pi*3)/8 and angle<(np.pi*5)/8 or (angle<-3*(np.pi/8) and angle>=-5*((np.pi)/8)): # 상 하 방향을 봅니다. 
                if G[row,col]>max(G[row+1,col], G[row-1,col]) : # 이웃들보다 크다면,
                    res[row,col]=G[row,col] # 값을 선택합니다.
            elif angle>=(np.pi*5)/8 and angle<(np.pi*7)/8 or (angle<-1*(np.pi/8) and angle>=-1*((np.pi*3)/8)): # 5시와 11시 방향을 봅니다.
                if G[row,col]>max(G[row-1,col-1], G[row+1,col+1]) : # 이웃들보다 크다면,
                    res[row,col]=G[row,col] # 값을 선택합니다.
            else : # 좌 우 방향을 봅니다.
                if G[row,col]>max(G[row,col-1], G[row,col+1]) : # 이웃들보다 크다면,
                    res[row,col]=G[row,col] # 값을 선택합니다.
            
    return res

def double_thresholding(img):
    """ 
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: double_thresholded image.
    """
    res=img.copy()
    res.fill(0) # 0으로 채운 새로운 배열을 선언합니다.
    max_img, min_img = np.max(img), np.min(img) # img intensity의 max, min값을 찾습니다.
    diff=max_img-min_img 
    t_high=min_img+0.15*diff
    t_low=min_img+0.03*diff # threshold max, min을 지정합니다.
    height=len(img)
    width=len(img[0])
    # lambda function을 사용해서 t_high보다 크다면 255, t_low보다 작다면 0, 그 사이라면 80으로 mapping해줍니다. 
    threshold_function = lambda img, t_high, t_low: 255 if img > t_high else (0 if img < t_low else 80) 
    for row in range(height) :
        for col in range(width) :
            res[row,col]=threshold_function(img[row,col],t_high,t_low) # mapping

    return res

def dfs(img, res, i, j, visited=[]):
    # calling dfs on (i, j) coordinate imply that
    #   1. the (i, j) is strong edge
    #   2. the (i, j) is weak edge connected to a strong edge
    # In case 2, it meets the condition to be a strong edge
    # therefore, change the value of the (i, j) which is weak edge to 255 which is strong edge
    res[i, j] = 255

    # mark the visitation
    visited.append((i, j))

    # examine (i, j)'s 8 neighbors
    # call dfs recursively if there is a weak edge
    for ii in range(i-1, i+2) :
        for jj in range(j-1, j+2) :
            if (img[ii, jj] == 80) and ((ii, jj) not in visited) :
                dfs(img, res, ii, jj, visited)

def hysteresis(img):
    """ Find weak edges connected to strong edges and link them.
    Iterate over each pixel in strong_edges and perform depth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: hysteresised image.
    """
    #implement 
    res=img.copy()
    res.fill(0) # 0으로 채운 새로운 배열을 선언합니다.
    
    for row in range(len(img)) :
        for col in range(len(img[0])) :
            if img[row,col]==255 : # strong edge 기준으로 dfs합니다. -> 주변 weak edge를 강화합니다.
                dfs(img, res, row, col, [])
    
    
    return res

def main():
    
    RGB_img = Image.open('./hw3/iguana.bmp')
    noise_reduced_img = reduce_noise(RGB_img)
    Image.fromarray(noise_reduced_img.astype('uint8')).save('./hw3/iguana_blurred.bmp', 'BMP')
    
    g, theta = sobel_filters(noise_reduced_img)
    Image.fromarray(g.astype('uint8')).save('./hw3/iguana_sobel_gradient.bmp', 'BMP')
    Image.fromarray(g.astype('uint8')).show()
    Image.fromarray(theta.astype('uint8')).save('./hw3/iguana_sobel_theta.bmp', 'BMP')
    Image.fromarray(theta.astype('uint8')).show()
    non_max_suppression_img = non_max_suppression(g, theta)
    Image.fromarray(non_max_suppression_img.astype('uint8')).save('./hw3/iguana_non_max_suppression.bmp', 'BMP')
    Image.fromarray(non_max_suppression_img.astype('uint8')).show()
    double_threshold_img = double_thresholding(non_max_suppression_img)
    Image.fromarray(double_threshold_img.astype('uint8')).save('./hw3/iguana_double_thresholding.bmp', 'BMP')
    Image.fromarray(double_threshold_img.astype('uint8')).show()
    hysteresis_img = hysteresis(double_threshold_img)
    Image.fromarray(hysteresis_img.astype('uint8')).save('./hw3/iguana_hysteresis.bmp', 'BMP')
    Image.fromarray(hysteresis_img.astype('uint8')).show()

main()