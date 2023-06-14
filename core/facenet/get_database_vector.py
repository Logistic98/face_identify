# -*- coding: utf-8 -*-

# 将所有底库中的图片进行加载

import cv2
from core.facenet.start_facenet import *

MARGIN = 0
MIN_SIZE = 30
FACE_PROBABILITY = 0.45
FRONT_BACK_OFFSET = 1


# 截取带比例margin的人脸
def get_margin_img(img, location):
    # 计算外扩margin
    margin_rate = 0.2
    margin_width = int(location['width'] * margin_rate)
    margin_height = int(location['height'] * margin_rate)
    max_width = img.shape[1]
    max_height = img.shape[0]
    # 判断会不会超边缘
    top = 0
    bottom = max_height
    left = 0
    right = max_width
    if (location['top'] - margin_height) > 0:
        top = location['top'] - margin_height
    if (location['top'] + location['height'] + margin_height) < max_height:
        bottom = location['top'] + location['height'] + margin_height
    if (location['left'] - margin_width) > 0:
        left = location['left'] - margin_width
    if (location['left'] + location['width'] + margin_width) < max_width:
        right = location['left'] + location['width'] + margin_width
    margin_img = img[top: bottom, left: right, :]
    return margin_img


def cal_img_database(detect_img):
    detect_img = cv2.resize(detect_img, (160, 160))
    v = get_face_vector(detect_img)
    return v


def get_all_basedata(data_face):

    '''
    1.遍历人员底库
    2.遍历所有的人员图片
    3.计算所有人员的vetor信息
    4.结果以json格式返回
    :param data_face: 人员底库路径
    '''

    # 记录所有的人员信息
    dic_person = dict()

    # 读取所有的人员底库文件夹
    data_person_list = os.listdir(data_face)

    # 遍历所有的文件夹，读取图片，计算图片的vector
    for data_person in data_person_list:

        data_person_path = os.path.join(data_face, data_person)
        img_list = os.listdir(data_person_path)
        dic_person[data_person] = []
        person_path_temp = os.path.join(data_face, data_person)

        for img_name in img_list:
            img_path_temp = os.path.join(person_path_temp, img_name)
            img_content = cv2.imread(img_path_temp)
            result = cal_img_database(img_content)
            dic_person[data_person].append(result)

    return dic_person

