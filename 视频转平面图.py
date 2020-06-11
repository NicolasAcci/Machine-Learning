import time
import cv2
import os

file_path = "F:\\VIDEO"
for root, dirs, files in os.walk(file_path):
    for file in files:
        # 获取文件所属目录
        video_path = root + "/" + file
        print(video_path)
        file_name = file.split(".")[0]
        print(file_name)
        # 获取文件路径
        # print(os.path.join(root,file))
        # video_path = "F:\\VIDEO\\20170101_000007.MP4"
        vid = cv2.VideoCapture(video_path)
        label, frame = vid.read()

        count = 1
        while label:
            if count % 90 == 0:
                cv2.imwrite("F:\\picture/" + file_name + "_" + str(count) + ".jpg", frame)
                t0 = time.clock()
                print("count:{}".format(count))
                print('time : ', (1 / (time.clock() - t0)) / 6000, "s")
                print(file_path)
            count += 1
            label, frame = vid.read()
        print("==================over=======================")
