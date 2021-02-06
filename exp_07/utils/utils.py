import os
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import face_recognition

def get_cropped_face(image_file):
    '''
    이미지에서 얼굴영역을 잘라오는 함수이다.
    input:
        image_file: 이미지가 들어있는 경로주소
    return:
        cropped_face : 얼굴영역을 검출하여 cropped 한 이미지
    '''
    image = face_recognition.load_image_file(image_file)
    face_locations = face_recognition.face_locations(image)
    try:
        a, b, c, d = face_locations[0]
        cropped_face = image[a:c,d:b,:]
        return cropped_face
    except:
        return False

def get_face_embedding(face):
    '''
    얼굴영역 임베딩벡터를 구하는 함수
    '''
    return face_recognition.face_encodings(face)

def get_face_embedding_dict(dir_path):
    '''
    param dir_path: 이미지 폴더 경로
    return:
        ebedding_dict : 이미지파일 이름을 키로 하고 value에 임베딩 벡터값을 담고 있는
                        딕셔너리 타입으로 리턴
    '''
    file_list = os.listdir(dir_path)
    embedding_dict = {}

    for file in file_list:
        image_path = os.path.join(dir_path, file)
        face = get_cropped_face(image_path)
        try:
             embedding = get_face_embedding(face)
        except:
            pass

        if len(embedding) > 0:
            embedding_dict[os.path.splitext(file)[0]] = embedding[0]

    return embedding_dict

def get_distance(name1, name2):
    '''
    단순히 두 벡터 사이 거리차이
    트리플 로스는 아님
    '''
    with open("embedding_dict.pkl", "rb") as f:
        embedding_dict = pickle.load(f)
    return np.linalg.norm(embedding_dict[name1]-embedding_dict[name2], ord=2)

def get_sort_key_func(name1):
    '''
    param name1: 비교 기준이 될 얼굴의 이름
    return: get_distance_from_name1
        이후 이 함수의 리턴값에 name2를 넣어주면 get_distance함수가 호출됨
    '''
    def get_distance_from_name1(name2):
        return get_distance(name1, name2)
    return get_distance_from_name1


def get_nearest_face(name, top=4):
    '''
    param name: 비교 기준이 될 얼굴의 이름
    param top: 임베딩 벡터 거리가 가까운 상위 몇 명까지 볼 것인가
    :return: 
             본인 제외 Top 만큼의 사람들이 나온다.
             top_embedding_matrix : top들의 임베딩 벡터를 dict형태로 돌려줌
    '''
    with open("embedding_dict.pkl", "rb") as f:
        embedding_dict = pickle.load(f)

    top_embedding_matrix = {}
    sort_key_func = get_sort_key_func(name)
    sorted_faces = sorted(embedding_dict.items(), key=lambda x: sort_key_func(x[0]))

    for i in range(top + 1):
        #if i == 0:
            #continue
        if sorted_faces[i]:
            print('순위 {}: 이름({}), 거리({})'.format(i, sorted_faces[i][0],
                                                 sort_key_func(sorted_faces[i][0])))
            top_embedding_matrix[sorted_faces[i][0]] = embedding_dict[sorted_faces[i][0]]

    return top_embedding_matrix
