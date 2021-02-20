import os
import urllib
import tarfile
from glob import glob
from os.path import join

import cv2
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt

def read_img(img_path, plot_img = False):
    '''
    input :
        img_path : 이미지 경로
        plot_img : 이미지 출력
    output :
        이미지 사이즈와 이미지
    return :
        img_orig
    '''
    img_orig = cv2.imread(img_path)
    
    print(img_orig.shape)
    
    if plot_img == True : plt.imshow(img_orig)
    
    return img_orig

def download_weight():
    # define model and download & load pretrained weight
    _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'

    model_dir = os.getcwd()
    tf.io.gfile.makedirs(model_dir)

    print ('temp directory:', model_dir)

    download_path = os.path.join(model_dir, 'deeplab_model.tar.gz')
    if not os.path.exists(download_path):
        urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + 'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
                    download_path)
    return download_path

def get_interest_obj(img, seg_map, obj_num = 15, plot_img=True):
    '''
    input:
        img_show : object를 찾아서 색칠할 사진
        seg_map : 픽셀이 가지는 값이 class index인 일종의 픽셀별 예측결과
        obj_num : 찾고 싶은 class의 label index(ex 15는 사람)
    output:
        img_mask : obj_num인 위치는 255, 나머지(back ground)는 0인 이미지
        img_show : object를 원본 이미지에서 다른 색으로 색칠하여 나타냄
    '''
    img_show = img.copy() #이미지 복사
    
    # 예측 중 선택한 class값이면 그 값으로, 아니면 0으로 죽임
    seg_map = np.where(seg_map == obj_num, obj_num, 0) # seg_map이 obj_num이면 obj_num으로 나머진 0으로 변환
    img_mask = seg_map * (255/seg_map.max()) # 255 normalization : 즉 값이 15인 픽셀들이 255(흰색)로 값이 바뀐다.
    img_mask = img_mask.astype(np.uint8)

    # 원본 이미지에서 object 위치에 색칠하기
    color_mask = cv2.applyColorMap(img_mask, cv2.COLORMAP_JET)
    plt.imshow(color_mask)
    img_show = cv2.addWeighted(img_show, 0.6, color_mask, 0.35, 0.0) # (a, 1, b, 3, 5) : (1*a) + (3*b) + 5

    if plot_img == True:
        plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
        plt.show()

    return img_mask, img_show

def resize_image(img_mask, img_orig, is_segment=True, plot_img=True):
    '''
    input :
        img_mask : 전처리가된 obj와 배경이 흰색과 검정색으로 구분된 이미지
        img_orig : 원본 이미지
    output :
        img_mask_resized : 사이지가 원본의 것으로 변경된 img_bg
    '''
    # 배경이될 사진의 사이즈를 원본 사이즈에 맞춰준다.
    img_mask_resized = cv2.resize(img_mask, img_orig.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
    
    if is_segment:
        _, img_mask_resized = cv2.threshold(img_mask_resized, 128, 255, cv2.THRESH_BINARY)

    if plot_img == True:
        ax = plt.subplot(1,2,1)
        plt.imshow(img_mask, cmap=plt.cm.binary_r)
        ax.set_title('Original Size Mask')

        ax = plt.subplot(1,2,2)
        plt.imshow(img_mask_resized, cmap=plt.cm.binary_r)
        ax.set_title('DeepLab Model Mask')

        plt.show()
        
    return img_mask_resized


def get_background(img_orig, img_mask, plot_img=True):
    '''
    input :
        img_orig : 원본 이미지
        img_mask : 원본 이미지와 사이즈가 같은 obj와 배경이 구분된 이미지
    output :
        img_bg : 원본에서 obj를 (0으로) 지우고, 배경만 남긴 이미지
    '''
    # 3채널로 바꿔준다.
    img_mask_color = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
    img_bg_mask = cv2.bitwise_not(img_mask_color) # 색반전 (obj(255->0), bg(0->255))
    img_bg = cv2.bitwise_and(img_orig, img_bg_mask) # 겹치는 부분만 살리고 다르면 없애기(1,1 => 1, 1,0 => 0)
    if plot_img == True:
        plt.imshow(img_bg)
        plt.show()
    
    return img_bg

def get_blur_image(img_bg, ksize=(100,100),plot_img=True):
    img_bg_blur = cv2.blur(img_bg, ksize)
    if plot_img == True:
        plt.imshow(cv2.cvtColor(img_bg_blur, cv2.COLOR_BGR2RGB))
        plt.show()
    
    return img_bg_blur

def concat_obj_bg(img_mask_re, img_orig, img_bg):
    if len(img_mask_re.shape) == 2:
        img_mask_color = cv2.cvtColor(img_mask_re, cv2.COLOR_GRAY2BGR)
    else:
        img_mask_color = img_mask_re
    # np.where(a,b,c) : a조건이 true인 b의 요소에 c를 적용
    img_concat = np.where(img_mask_color==255, img_orig, img_bg)
    plt.imshow(cv2.cvtColor(img_concat, cv2.COLOR_BGR2RGB))
    plt.show()