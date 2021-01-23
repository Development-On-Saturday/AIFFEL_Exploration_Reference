import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import dlib
import imutils


def getimage(file_path, img_name):
    '''
    이미지가 있는 폴더 경로와, 파일명을 인자로 넣어준다.
    return:
        img_show : rgb 이미지
        img_bgr : bgr 이미지
    '''
    img_path = file_path
    img_bgr = cv2.imread(img_path+img_name)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_show = img_rgb.copy()

    plt.imshow(img_show)
    plt.show()

    return img_show, img_bgr

def img_resize(img, width, height):
    '''
    원하는 사이즈로 이미지를 재조정할 수 있다.
    return:
        img_re : resized image
    '''
    img_re = cv2.resize(img, (width, height))
    plt.imshow(img_re)
    plt.show()
    return img_re

def face_detector(img, pyramid=1):
    '''
    이미지를 넣어주면 얼굴영역 잡아낸다.
    return:
        img : 입력이미지의 얼굴 영역에 bbox가 입혀진 이미지
        dlib_rects : bbox 좌표
    '''
    detector_hog = dlib.get_frontal_face_detector()
    dlib_rects = detector_hog(img, pyramid)

    for dlib_rect in dlib_rects:
        l = dlib_rect.left()
        t = dlib_rect.top()
        r = dlib_rect.right()
        b = dlib_rect.bottom()

        cv2.rectangle(img, (l,t), (r,b), (0,255,0), 2, lineType=cv2.LINE_AA)

    #plt.imshow(img)
    #plt.show()

    return img, dlib_rects

def get_landmark(img, dlib_rects):
    '''
    dlib이 제공하는 얼굴 ladnmark 추출기는 특징점 68개를 찾아준다.
    landmark_predictor : 랜드마크 68개의 점들이 튜플처럼 (x,y) 순서로 points 안에 들어가 있고
    이 점들의 (x,y)를 우리가 알고 있는 리스트의 형태로 변환시킨다음 list_landmarks에 어팬드 시켜준다.
    return:
        list_landmarks: 얼굴마다 좌표정보를 담고 있다.
        lsit_points :
        img_rgb : bbox와 특징점이 입혀진 이미지
    '''
    model_path = os.getenv('HOME') + '/project/E10/models/shape_predictor_68_face_landmarks.dat'
    landmark_predictor = dlib.shape_predictor(model_path)

    list_landmarks = []
    for dlib_rect in dlib_rects:
        points = landmark_predictor(img, dlib_rect)
        list_points = list(map(lambda p:(p.x, p.y), points.parts()))
        list_landmarks.append(list_points)

    for landmark in list_landmarks:
        for idx, point in enumerate(landmark):
            cv2.circle(img, point, 2, (0,255,255), -1)
    plt.imshow(img)
    plt.show()

    return list_landmarks, list_points, img

def color_reverse(img, show_img = True):
    '''
    색상을 반전시켜주는 함수이다.
    스티커 이미지를 생상반전 없이 회전시켰을때 네 모서리에 생기는 검정색 영역을
    방지하기 위해서 이다.
    즉 색상 반전 -> 회전 -> 재반전 -> 이미지 위에 붙이기 과정을 다른다.
    return:
        img_re : 색상이 반전된 이미지
    '''

    img_re = cv2.bitwise_not(img)
    if show_img == True:
        plt.imshow(img_re)
        plt.show()
    return img_re

def set_sticker(dlib_rects, list_landmarks, sticker_img, discribe = True):
    '''
    bbox와 특징점 정보와 스티커 이미지를 인자로 준다.
    return:
        refined_x : 스티커를 붙일 원본 이미지 위 x 좌표 위치
        refined_y : 스티커를 붙일 원본 이미지 위 y 좌표 위치
        sticker_img : 사이즈 조정, 회전 된 스티커 이미지
    '''
    stickers = []
    refined_xs = []
    refined_ys = []

    for dlib_rect, landmark in zip(dlib_rects, list_landmarks):
        x = landmark[30][0]
        y = landmark[30][1] + ((landmark[33][1]-landmark[30][1]) // 2)
        w = (landmark[14][0] - landmark[2][0])
        h = w

        v = np.array([[landmark[2][0]], [0]])
        u = np.array([[landmark[14][0]], [landmark[14][1]]])
        product = np.dot(v.T,u)
        length = np.sqrt(np.dot(v.T,v)) * np.sqrt(np.dot(u.T,u))
        theta = int(np.degrees(np.arccos(product/length)))

        if discribe == True:
            print('(x,y):(%d,%d)'%(x,y))
            print('(w,h):(%d,%d)'%(w,h))
            print('rotation : (%d)'%(theta))

        sticker_re = cv2.resize(sticker_img, (w,h))
        sticker_rot = imutils.rotate(sticker_re,-1*theta)
        sticker = color_reverse(sticker_rot, show_img = False)

        stickers.append(sticker)

        if (x - w//2) < 0:
            refined_xs.append(0)
        elif (x - w//2) >=0:
            refined_xs.append(x - w//2)

        if (y - h//2) < 0:
            refined_ys.append(0)
        elif (y - h//2) > 0:
            refined_ys.append(y - h//2)



    return refined_xs, refined_ys, stickers

def attach_sticker(img, stickers, refined_xs, refined_ys, show_img = True):
    for refined_x, refined_y, sticker in zip(refined_xs, refined_ys, stickers):
        sticker_area = img[refined_y:refined_y+sticker.shape[0],
                    refined_x:refined_x+sticker.shape[1]]
        img[refined_y:refined_y+sticker.shape[0],
            refined_x:refined_x+sticker.shape[1]] = np.where(sticker == 255,
                                                            sticker_area, sticker).astype(np.uint8)
    if show_img == True:
        plt.imshow(img)
        plt.show()
    else:
        pass
