## 将RetinaFace和FaceNet封装成特定人物识别服务

### 1. 目录结构及使用说明

使用 Flask 将 RetinaFace 和 FaceNet 封装成特定人物识别服务。

项目目录结构如下：

```
.
├── core    // 核心算法模块（包括 RetinaFace 和 FaceNet 及相关处理）
│   ├── data
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── data_augment.py
│   │   └── wider_face.py
│   ├── facenet
│   │   ├── facenet.py
│   │   ├── get_database_vector.py
│   │   └── start_facenet.py
│   ├── layers
│   │   ├── __init__.py
│   │   ├── functions
│   │   └── modules
│   ├── models   // 算法模型
│   │   ├── Retinaface_model_v2
│   │   ├── facenet
│   │   ├── net.py
│   │   └── retinaface.py
│   ├── save_face_database  // 人脸底库
│   │   ├── baideng
│   │   └── telangpu
│   ├── utils
│   │   ├── box_utils.py
│   │   ├── compare_vector.py
│   │   ├── nms
│   │   └── timer.py
│   └── face_identify.py  // 人脸识别服务的主入口（用于给Flask调用）
├── log.py
├── code.py
├── response.py
├── server.py         // 使用Flask封装好的服务
└── server_test.py    // 测试服务
```

使用说明：

- [1] 运行环境：项目代码里我使用的是CPU运行方式。如果有GPU，可以修改一下配置，将其改成GPU方式运行（需要注意的是，PyTorch 及 Tensorflow 的依赖需要安装对应GPU版本的）。

- [2] 人脸底库更换：人脸底库放在 save_face_database 目录，制作人脸底库的时候注意只保留人脸，左侧脸30度+右侧脸30度+正脸各2张，更换人脸底库后，需要修改一下 face_identify.py 程序的配置，将如下配置改成自己的。

  ```python
  database_path = 'core/save_face_database'
  vector_database = get_all_basedata(database_path)
  dic_info_person = {}
  dic_info_person['baideng'] = ['拜登']
  dic_info_person['telangpu'] = ['特朗普']
  ```

### 2. 测试运行特定人物识别的服务

安装好依赖之后，先启动 server.py 服务，再执行 server_test.py 即可本地测试。

```
$ python3 server.py 
$ python3 server_test.py
```

### 3. 服务器上使用Docker部署服务 

服务器环境：Debian 11 x86_64 系统，8GB内存，160GB存储，2x Intel Xeon CPU，无GPU，带宽1 Gigabit，Docker version 20.10.17

部署测试：将整个项目上传到服务器上，执行 install.sh 安装脚本，部署成功后改一下 server_test.py 的 IP 对其进行功能测试。正式部署使用nohup后台运行。

```
$ chmod u+x install.sh && ./install.sh
$ docker exec -it face_identify /bin/bash
$ python server.py
```

