# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse

from core.facenet import facenet
from core.layers.functions.prior_box import PriorBox
from core.utils.nms.py_cpu_nms import py_cpu_nms
from core.facenet.get_database_vector import *
from core.utils.compare_vector import *
import numpy as np
import time
from core.models.retinaface import RetinaFace
from core.utils.box_utils import decode, decode_landm
from core.data import cfg_mnet, cfg_re50
import torch
from numpy.core import shape

model_path_up = './models/facenet'


def get_face_vector(img):
    start_time = time.time()
    result_vector = facenet.run(img)
    end_time = time.time()
    print('recognize time is:', end_time-start_time)
    return result_vector


database_path = 'core/save_face_database'
vector_database = get_all_basedata(database_path)
dic_info_person = {}
dic_info_person['baideng'] = ['拜登']
dic_info_person['telangpu'] = ['特朗普']
# print(dic_info_person)


parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='./core/models/Retinaface_model_v2/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu',  default=True, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.95, type=float, help='visualization_threshold')
parser.add_argument('--video', default='', type=str, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(0))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def cal_img(img):
    '''
    计算小图人脸的 vector信息
    :param img:
    :return:
    '''
    detect_img = cv2.resize(img, (160, 160))
    v = get_face_vector(detect_img)
    return v  # 边框信息和 vector


def compare_vector(vec_database, vec_img):
    '''
    底库人脸vector；视频中人脸vector
    :param vec_database: 底库人脸vector
    :param vec_img: 视频中人脸vector
    :return:
    '''
    is_get, label_result = get_max_label(vec_img, vec_database)
    if is_get:
        result_dic = label_result
    else:
        result_dic = ''
    return result_dic


torch.set_grad_enabled(False)
cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50
# net and model
net = RetinaFace(cfg=cfg, phase = 'test')
net = load_model(net, args.trained_model, args.cpu)
net.eval()
print('Finished loading model!')
print(net)
device = torch.device("cpu" if args.cpu else "cuda")
net = net.to(device)
MARGIN = 0
MIN_SIZE = 30
FACE_PROBABILITY = 0.95
FRONT_BACK_OFFSET = 1


def face_recognize(image):
    img_raw = image
    img = np.float32(img_raw)
    im_height, im_width, _ = img.shape
    with torch.no_grad():
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)

        img = img.to(device)
        scale = scale.to(device)
        resize = 1
        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))

        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms_a = landms.cpu().numpy()
        del landms
        torch.cuda.empty_cache()

        landms=landms_a

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        for b in dets:

            if b[4] < args.vis_thres:
                continue
            b = list(map(int, b))

            # 人脸图片
            if b[1] < 0:
                b[1] = 0
            if b[3] > shape(img_raw)[0]:
                b[3] = shape(img_raw)[0]
            if b[0] < 0:
                b[0] = 0
            if b[2] > shape(img_raw)[1]:
                b[2] = shape(img_raw)[1]
            face_img = img_raw[b[1]:b[3], b[0]:b[2]]

            vector_info = cal_img(face_img)
            result_label = compare_vector(vector_database, vector_info)

            if result_label != '':
                global dic_info_person
                face_identify_result = dic_info_person[result_label]
                return face_identify_result


def img_face_identify(img_path):
    image = cv2.imread(img_path)
    face_identify_result = face_recognize(image)
    return face_identify_result


if __name__ == '__main__':
    img_path = 'save_face_database/baideng/baideng001.jpg'
    image = cv2.imread(img_path)
    face_identify_result = face_recognize(image)
    print(face_identify_result)