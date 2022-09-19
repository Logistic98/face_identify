# -*- coding: utf-8 -*-

import os
from uuid import uuid1
from flask import Flask, jsonify
from flask_cors import CORS
from pre_request import pre, Rule
import base64

from code import ResponseCode, ResponseMessage
from core.face_identify import img_face_identify
from log import logger


# 创建一个服务
app = Flask(__name__)
CORS(app, supports_credentials=True)


# 解析base64生成图像文件
def base64_to_img(image_b64, img_path):
    imgdata = base64.b64decode(image_b64)
    file = open(img_path, 'wb')
    file.write(imgdata)
    file.close()


"""
# 图片人脸识别--识别人脸底库中的特定人脸
"""
@app.route(rule='/api/faceIdentify/identifySpecificFaces', methods=['POST'])
def identifySpecificFaces():

    # 参数校验
    rule = {
        "img": Rule(type=str, required=True),
        "ext": Rule(type=str, required=False)
    }
    try:
        params = pre.parse(rule=rule)
    except Exception as e:
        logger.error(e)
        fail_response = dict(code=ResponseCode.RARAM_FAIL, msg=ResponseMessage.RARAM_FAIL, data=None)
        logger.error(fail_response)
        return jsonify(fail_response)

    # 获取参数
    image_b64 = params.get("img")
    ext = params.get("ext")

    # 将base64字符串解析成图片保存
    if not os.path.exists('./img'):
        os.makedirs('./img')
    uuid = uuid1()
    if ext is not None:
        img_path = './img/{}.{}'.format(uuid, ext)
    else:
        img_path = './img/{}.jpg'.format(uuid)
    try:
        base64_to_img(image_b64, img_path)
    except Exception as e:
        logger.error(e)
        fail_response = dict(code=ResponseCode.BUSINESS_FAIL, msg=ResponseMessage.BUSINESS_FAIL, data=None)
        logger.error(fail_response)
        return jsonify(fail_response)

    # 对图片对比人脸底库中的特定人脸进行识别
    try:
        result = img_face_identify(img_path)
    except Exception as e:
        logger.error(e)
        os.remove(img_path)
        fail_response = dict(code=ResponseCode.BUSINESS_FAIL, msg=ResponseMessage.BUSINESS_FAIL, data=None)
        logger.error(fail_response)
        return jsonify(fail_response)

    # 处理完成后删除生成的图片文件
    os.remove(img_path)

    # 成功的结果返回
    success_response = dict(code=ResponseCode.SUCCESS, msg=ResponseMessage.SUCCESS, data=result)
    logger.info(success_response)
    return jsonify(success_response)


if __name__ == '__main__':
    # 解决中文乱码问题
    app.config['JSON_AS_ASCII'] = False
    # 启动服务 指定主机和端口
    app.run(host='0.0.0.0', port=5007, debug=False, threaded=True)