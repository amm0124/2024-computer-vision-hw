from PIL import Image
import numpy as np
import math

# part1-1 : boxfilter 만들기
def boxfilter(n) :
    if n%2==0 :
        raise AssertionError('Dimension must be odd') 
    else :
        box_filter=[ [1/(n**2) for _ in range(n)] for _ in range(n)]
        return np.array(box_filter)   
    
#print(boxfilter(n=3))
#boxfilter(n=4)
#print(boxfilter(n=7))
    
# part1-2 : 1차원 gaussian filter 만들기 
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

print("sigma=0.3 " ,gauss1d(sigma=0.3))
print("sigma=0.5 " ,gauss1d(sigma=0.5))
print("sigma=1 " ,gauss1d(sigma=1))
print("sigma=2 " ,gauss1d(sigma=2))

# part1-3 : 2차원 gaussian filter 만들기 
def gauss2d(sigma) : 
    # sigma값을 입력으로, 1차원 gaussian filter를 만들고, np.outer과 np.transpose를 통해 2차원 gaussian filter를 만듭니다.
    # np.outer후 결과값의 합은 어차피 1일 것이므로, 따로 정규화해줄 필요는 없습니다.
    # 왜냐하면, 모든 원소의 합이 1인 이미 정규화된 1차원 gaussain filer에, transpose후 outer product한 
    # 2차원 gaussian filer의 모든 원소의 합은 1일 것이기 때문입니다.
    array_1d=gauss1d(sigma)
    return np.outer(array_1d, np.transpose(array_1d)) 

print("sigma=0.5 " , gauss2d(sigma=0.5))
print("sigma=1 " , gauss2d(sigma=1))

# part1-4-(a) : array(=image)와 filter를 사용해서 image convolution 하기
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

# part1-4-(b) : sigma에 해당하는 2d gaussian filter를 만든 후, array(=image)와 convolution  
def gaussconvolve2d(array,sigma) :
    array=array.astype(np.float32)
    gaussian_filter=gauss2d(sigma)
    return convolve2d(array, gaussian_filter)
    
# part1-4-(c), (d) : sigma=3으로 tiger gaussian convolution하기.
image1 = Image.open('./hw2/images/3b_tiger.bmp') 
image1.show()
image1=np.asarray(image1.convert('L')) # 흑백으로 전환 후, nparray로 변환
convolution_tiger=gaussconvolve2d(image1, 3).astype('uint8') # gaussian convolution후, 정수 값으로 변환
tiger = Image.fromarray(convolution_tiger) 
tiger.show() 
tiger.save('part1-4-after.png' , 'PNG') #저장

# 새로운 실험 : boxfilter과 gaussian filter 차이
"""image2=Image.open('./hw2/images/2a_mangosteen.bmp')
r,g,b=image2.split() # split method를 사용해서 간단하게 rgb channel을 추출할 수 있습니다.
image2_r, image2_g, image2_b = np.asarray(r) , np.asarray(g), np.asarray(b)
image2_box_r=convolve2d(image2_r, boxfilter(11))
image2_box_g=convolve2d(image2_g, boxfilter(11))
image2_box_b=convolve2d(image2_b, boxfilter(11))

image2_result_r, image2_result_g, image2_result_b = image2_box_r.astype('uint8'), image2_box_g.astype('uint8'), image2_box_b.astype('uint8')
new_r, new_g, new_b = Image.fromarray(image2_result_r), Image.fromarray(image2_result_g), Image.fromarray(image2_result_b)
new_image2 = Image.merge('RGB', (new_r, new_g, new_b))
new_image2.show()"""

# part2-1 : gaussian filter(=low pass filter) 를 사용해서 image blurring하기. 
# image의 high-frequency를 제거하는 효과와 같습니다. (=low-frequency만 남기는 효과와 같습니다.)
# low-frequency만 남기는 이미지로 mangosteen을 선택했습니다.

image2=Image.open('./hw2/images/2a_mangosteen.bmp')
image2.show()
r,g,b=image2.split() # split method를 사용해서 간단하게 rgb channel을 추출할 수 있습니다.
image2_r, image2_g, image2_b = np.asarray(r) , np.asarray(g), np.asarray(b)

# r,g,b channel을 sigma=2인 2d-gaussian filter과 convolution합니다.
image2_convolved_r=gaussconvolve2d(image2_r, 2)
image2_convolved_g=gaussconvolve2d(image2_g, 2)
image2_convolved_b=gaussconvolve2d(image2_b, 2)

# unit8 type으로 변환 후, merge method를 통해, low-frequency image를 생성합니다.
image2_result_r, image2_result_g, image2_result_b = image2_convolved_r.astype('uint8'), image2_convolved_g.astype('uint8'), image2_convolved_b.astype('uint8')
new_r, new_g, new_b = Image.fromarray(image2_result_r), Image.fromarray(image2_result_g), Image.fromarray(image2_result_b)
new_image2 = Image.merge('RGB', (new_r, new_g, new_b))
new_image2.show()
#new_image2.save('part2-1-after.png' , 'PNG') 


# part2-2 : high-frequency image 만들기.
# 원래 image에서 low-frequency를 뺀 결과는 high-frequency image입니다.
# high frequency만 남기는 이미지로 orange를 선택했습니다.

image3=Image.open('./hw2/images/2b_orange.bmp')
image3.show()
r,g,b=image3.split() # split method를 사용해서 간단하게 rgb channel을 추출할 수 있습니다.

# r,g,b channel을 sigma=2인 2d-gaussian filter과 convolution합니다.
image3_r, image3_g, image3_b = np.asarray(r), np.asarray(g), np.asarray(b)
blur_r = gaussconvolve2d(image3_r, 2)
blur_g = gaussconvolve2d(image3_g, 2)
blur_b = gaussconvolve2d(image3_b, 2)

# 원본 이미지에서 gaussian filter와 convolution된 이미지를 빼서, image의 high-frequency만 남깁니다.
image3_result_r = image3_r - blur_r
image3_result_g = image3_g - blur_g
image3_result_b = image3_b - blur_b

# 128을 더해서, 음수 값을 보정합니다.
add_result_r = image3_result_r + 128
add_result_g = image3_result_g + 128
add_result_b = image3_result_b + 128

# 255 초과인 값을 보정합니다.
modified_result_r = np.minimum(add_result_r, 255)
modified_result_g = np.minimum(add_result_g, 255)
modified_result_b = np.minimum(add_result_b, 255)

#modified_result_r=np.where(add_result_r>255, 255 ,add_result_r)
#modified_result_g=np.where(add_result_g>255, 255 ,add_result_g)
#modified_result_b=np.where(add_result_b>255, 255 ,add_result_b)

# unit8 type으로 변환 후, merge method를 통해, high-frequency image를 생성합니다.
modified_result_r, modified_result_g, modified_result_b = modified_result_r.astype('uint8'), modified_result_g.astype('uint8'), modified_result_b.astype('uint8')
new_r, new_g, new_b = Image.fromarray(modified_result_r), Image.fromarray(modified_result_g), Image.fromarray(modified_result_b)
new_image3 = Image.merge('RGB', (new_r, new_g, new_b))
new_image3.show()
#new_image3.save('part2-2-after.png' , 'PNG') 

# part2-3 : low-frequency image와 high-frequency image 합성하기.
# 기존에 구했던, low-frequency image와 high-frequency image를 더한 후, 값을 보정해서 만듭니다.

hybrid_r=image2_result_r + image3_result_r
hybrid_g=image2_result_g + image3_result_g
hybrid_b=image2_result_b + image3_result_b

# np.clip을 이용해서 값을 보정합니다.
hybrid_r_corrected = np.clip(hybrid_r, 0, 255)
hybrid_g_corrected = np.clip(hybrid_g, 0, 255)
hybrid_b_corrected = np.clip(hybrid_b, 0, 255)

# unit8 type으로 변환 후, merge method를 통해, hybrid image를 생성합니다.
hybrid_r_corrected, hybrid_g_corrected, hybrid_b_corrected = hybrid_r_corrected.astype('uint8'), hybrid_g_corrected.astype('uint8'), hybrid_b_corrected.astype('uint8')
new_r, new_g, new_b = Image.fromarray(hybrid_r_corrected), Image.fromarray(hybrid_g_corrected), Image.fromarray(hybrid_b_corrected)
new_image4 = Image.merge('RGB', (new_r, new_g, new_b))
new_image4.show()
new_image4.save('part2-3-after.png' , 'PNG') 