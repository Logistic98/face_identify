# -*- coding: utf-8 -*-

import numpy as np


def get_max_label(vec_one, vec_database):
    corr_result = 0
    label_vec = 0

    for item, value in vec_database.items():
        for value_iter in value:
            corr_result_temp = np.dot(vec_one, value_iter)
            if corr_result_temp > corr_result:
                corr_result = corr_result_temp
                label_vec = item
    if corr_result > 0.6:
        return True, label_vec
    else:
        return False, label_vec


def compare_vector(vec_database, vec_img):

    '''
    底库人脸vector；视频中人脸vector
    :param vec_database: 底库人脸vector
    :param vec_img: 视频中人脸vector
    :return:
    '''
    result_dic = {}

    for i, vec_img_iter in enumerate(vec_img):
        is_get, label_result = get_max_label(vec_img_iter, vec_database)

        if is_get:
            result_dic[i] = label_result
        else:
            result_dic[i] = ''
    return result_dic
