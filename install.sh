#!/bin/bash
docker build -t face_identify_image .
docker run -itd -p 5007:5007 --name face_identify -e TZ="Asia/Shanghai" face_identify_image:latest