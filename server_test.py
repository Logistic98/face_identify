# -*- coding: utf-8 -*-

import base64
import requests
import json


def server_test():
    # 测试请求
    url = 'http://{0}:{1}/api/faceIdentify/identifySpecificFaces'.format("127.0.0.1", "5007")
    f = open('./core/save_face_database/baideng/baideng001.jpg', 'rb')
    # base64编码
    base64_data = base64.b64encode(f.read())
    f.close()
    base64_data = base64_data.decode()
    # 传输的数据格式
    data = {"img": base64_data}
    # post传递数据
    headers = {'Content-Type': 'application/json'}
    r = requests.post(url, headers=headers, data=json.dumps(data))
    print(r.text)


if __name__ == '__main__':
    server_test()

