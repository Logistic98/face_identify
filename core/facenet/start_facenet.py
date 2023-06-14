# -*- coding: utf-8 -*-

from .facenet import TFSession, Facenet
import os

model_path_up = './core/models/facenet'

cuda = '1'  # 指定显卡
gpu_mem = 0.2  # 指定显存
model_name = '20180827_489_model_982_400'  # 模型名
sess = TFSession(cuda, gpu_mem).get_sess()
model_path = os.path.join(model_path_up, model_name)
facenet = Facenet(sess, model_path)


def get_face_vector(img):
    result_vector = facenet.run(img)
    return result_vector
