import os
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
import dlib

class FaceDetection:
    """
    Face detection 코드 모듈화
    얼굴을 인식하고 얼굴에 콧수염 사진을 붙히는 모듈
    AIFFEL Exploration 10
    OpenCV, dlib 사용
    - 이미지 로드
    - 바운딩 박스 만들기
    - 얼굴 랜드마크 찍기
    - 스티커 붙히기
    - 이미지 로테이션
    - 이미지 줌
    - 이미지 밝기 조정
    """
    image_path = ""
    sticker_path = ""
    def __init__(self, dlib_model_path):
        self.dlib_model_path = dlib_model_path 
        self.detector_hog = dlib.get_frontal_face_detector()
        
    def image_load(self):
        img_bgr = cv2.imread(self.image_path)
        h, w, _ = img_bgr.shape
        # 사진의 비율 보존
        if h-w < 0 :
            img_bgr = cv2.resize(img_bgr, (640, 480))
        else:
            img_bgr = cv2.resize(img_bgr, (480, 640))
        
        # # image print
        # plt.imshow(img_bgr)
        # plt.show()
        return img_bgr

    def __str__(self):
        temp = self.image_load()
        return  '{} , BGR'.format(temp.shape)

    def face_bounding_box(self, img, BGR2RGB=True):
        if BGR2RGB ==True:
            img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dlib_rects = self.detector_hog(img, 1)

        if len(dlib_rects) ==0:
            return '얼굴 인식에 실패하였습니다'
        else:
            for dlib_rect in dlib_rects:
                l = dlib_rect.left()
                t = dlib_rect.top()
                r = dlib_rect.right()
                b = dlib_rect.bottom()
                # box 
                cv2.rectangle(img, (l,t), (r,b), (0,255,0), 2, lineType=cv2.LINE_AA)
            plt.imshow(img)
            print(dlib_rects)
            plt.title('Face Bounding Box', fontsize=15)
            return img

    def _find_landmark(self, img):
        landmark_predictor = dlib.shape_predictor(self.dlib_model_path)
        dlib_rects = self.detector_hog(img, 1)
        if len(dlib_rects) ==0:
            return '얼굴 인식에 실패하였습니다'
        list_landmarks = []
        for dlib_rect in dlib_rects:
            points = landmark_predictor(img, dlib_rect)
            list_points = list(map(lambda p: (p.x, p.y), points.parts()))
            list_landmarks.append(list_points)
        return list_landmarks, list_points


    def face_landmark(self, img, BGR2RGB=True):        
        list_landmarks, list_points = self._find_landmark(img)

        for landmark in list_landmarks:
            for point in list_points:
                cv2.circle(img, point, 2, (0,255,255), -1)
                
        if BGR2RGB ==True:
            img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        print('The number of dots : {}'.format(len(landmark)))
        plt.title('Face Landmark', fontsize=15)
        plt.show()
        return img
    def attaching_sticker(self, img, sticker_pixel ,  BGR2RGB=True):
        if BGR2RGB ==True:
            img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_sticker = cv2.imread(self.sticker_path)
            img_sticker = cv2.cvtColor(img_sticker, cv2.COLOR_BGR2RGB)

        dlib_rects = self.detector_hog(img, 1)
        list_landmarks , _= self._find_landmark(img)
        if len(dlib_rects) ==0:
            return '얼굴 인식에 실패하였습니다'
        else:
            for dlib_rect, landmark in zip(dlib_rects, list_landmarks):
                # 코 위치에 수염 달기, 코 위치 33
                x = landmark[33][0]
                y = landmark[33][1]
                w = dlib_rect.width()
                h = dlib_rect.height()
            
            img_sticker = cv2.resize(img_sticker, (w,h))

            refined_x = x - img_sticker.shape[1] // 2 # left
            refined_y = y - img_sticker.shape[0] // 2 # top

            if refined_y < 0 :
                img_sticker = img_sticker[-refined_y:]
                refined_y = 0

            sticker_area = img[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
            img[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \
                np.where(img_sticker==sticker_pixel,sticker_area,img_sticker).astype(np.uint8)
            plt.imshow(img)
            plt.title('Face with sticker', fontsize=15)
            plt.axis('off')
            plt.show()
            return img

    def rotate_image(self, image, angle, save=False, BGR2RGB=True):
        if BGR2RGB ==True:
            image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        if save :
            print('저장경로 : ./rotate.jpg')
            cv2.imwrite('rotate.jpg', result)
        plt.imshow(result)
        plt.title('Rotated Image',fontsize=15)
        return result
    
    def mod_brightness(self,img, brightness=True, BGR2RGB=True):
        if BGR2RGB ==True:
            img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if brightness :
            m = np.ones(img.shape, dtype="uint8") * 100
            result = cv2.add(img, m)
        else :
            m = np.ones(img.shape, dtype="uint8") * 50
            result = cv2.subtract(img, m)
        plt.imshow(result)
        plt.title('Bright Image', fontsize=15)
        return result

    def paddedzoom(self,img, zoom,BGR2RGB=True):
        if BGR2RGB ==True:
            img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = np.zeros(img.shape)
        zoomed = cv2.resize(img, dsize=(0,0), fx = zoom, fy = zoom)
        
        h, w, _ = img.shape
        zh, zw, _ = zoomed.shape
        if zoom<1 :
            out[(h-zh)//2:-(h-zh)//2, (w-zw)//2:-(w-zw)//2] = zoomed
        elif zoom > 1:
            out = zoomed[(zh-h)//2 : -(zh-h)//2, (zw-w)//2:-(zw-w)//2]
        else :
            out = img
        out = out.astype(np.uint8)
        plt.imshow(out)
        plt.title('Zoom Image', fontsize=15)
        
        return out
    
    @classmethod
    def load_image(cls, image_path, BGR2RGB=True):
        cls.image_path = image_path
        img = cv2.imread(cls.image_path)
        if BGR2RGB ==True:
            img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    @classmethod
    def load_sticker(cls, sticker_image_path):
        cls.sticker_path = sticker_image_path
        return cls.sticker_path