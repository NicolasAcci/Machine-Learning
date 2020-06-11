import argparse
import json
import os
import os.path as osp
import warnings
import copy
import cv2
import numpy as np
import PIL.Image
from skimage import io
import yaml
from labelme import utils

# labelme 转换指定的label像素值
# 这个只能按照顺序来
NAME_LABEL_MAP = {
    '_background_': 0,
    "road": 1,
    "grass": 2,
    "building": 3,
    "car": 4,
    "person": 5,
    "tree": 6,
    "pavement": 7,
    "lamp": 8,
    "step": 9,
    "divide": 10,
    "lane": 11,
    "sign": 12,
    "pedestrian": 13,
    "motorcycle": 14,
    "water": 15,
    "roadblock": 16,
    "bicycle": 17,
}

# NAME_LABEL_MAP = {
#     '_background_': 0,
#     "road": 11,
#     "lane": 12,
#     "pedestrain": 13,
#     "sign": 14,
#     "pavement": 15,
#     "grass": 16,
#     "water": 17,
#     "building": 21,
#     "step": 22,
#     "tree": 23,
#     "lamp": 24,
#     "man": 31,
#     "car": 32
# }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', default='label_file/road_label')
    parser.add_argument('-o', '--out', default=None)
    args = parser.parse_args()
    # out_path = '/home/lingtan/data/lane_detection/Codes-for-Lane-Detection/SCNN-Tensorflow/lane-detection-model/label_file/data/'
    out_path = 'D:\\2\\ist\\all_more_list/'

    data_flag = '0515'

    json_file = out_path +'json/'
    print("json_file", json_file)
    list = os.listdir(json_file)
    print("list", list, len(list))
    for i in range(0, len(list)):
        try:
            path = os.path.join(json_file, list[i])
            print("path",path)
            filename = list[i][:-5]  # .json
            if os.path.isfile(path):
                data = json.load(open(path))
                img = utils.image.img_b64_to_arr(data['imageData'])
                lbl, lbl_names = utils.shape.labelme_shapes_to_label(img.shape,
                                                                     data['shapes'])  # labelme_shapes_to_label
                print("label1", lbl.shape, lbl_names)
                # print("label2", lbl)
                # modify labels according to NAME_LABEL_MAP
                lbl_tmp = copy.copy(lbl)
                for key_name in lbl_names:
                    # print("label1",key_name)
                    old_lbl_val = lbl_names[key_name]
                    # print("label2_old", old_lbl_val)
                    new_lbl_val = NAME_LABEL_MAP[key_name]
                    # print("label2_new", new_lbl_val)
                    # print("T1",key_name)
                    # 设定规定值的标签图像
                    if key_name == "building":
                        new_lbl_val = 3
                        # print("label2-1_new", new_lbl_val)
                        lbl_tmp[lbl == old_lbl_val] = new_lbl_val
                    elif key_name == "road":
                        new_lbl_val = 1
                        # print("label2-2_new", new_lbl_val)
                        lbl_tmp[lbl == old_lbl_val] = new_lbl_val
                    elif key_name == "grass":
                        new_lbl_val = 2
                        # print("label2-2_new", new_lbl_val)
                        lbl_tmp[lbl == old_lbl_val] = new_lbl_val
                    elif key_name == "car":
                        new_lbl_val = 4
                        # print("label2-3_new", new_lbl_val)
                        lbl_tmp[lbl == old_lbl_val] = new_lbl_val
                    elif key_name == "person":
                        new_lbl_val = 5
                        # print("label2-3_new", new_lbl_val)
                        lbl_tmp[lbl == old_lbl_val] = new_lbl_val
                    elif key_name == "tree":
                        new_lbl_val = 6
                        # print("label2-3_new", new_lbl_val)
                        lbl_tmp[lbl == old_lbl_val] = new_lbl_val
                    elif key_name == "pavement":
                        new_lbl_val = 7
                        # print("label2-3_new", new_lbl_val)
                        lbl_tmp[lbl == old_lbl_val] = new_lbl_val
                    elif key_name == "lamp":
                        new_lbl_val = 8
                        # print("label2-3_new", new_lbl_val)
                        lbl_tmp[lbl == old_lbl_val] = new_lbl_val
                    elif key_name == "step":
                        new_lbl_val = 9
                        # print("label2-3_new", new_lbl_val)
                        lbl_tmp[lbl == old_lbl_val] = new_lbl_val
                    elif key_name == "divide":
                        new_lbl_val = 10
                        # print("label2-3_new", new_lbl_val)
                        lbl_tmp[lbl == old_lbl_val] = new_lbl_val
                    elif key_name == "lane":
                        new_lbl_val = 11
                        # print("label2-3_new", new_lbl_val)
                        lbl_tmp[lbl == old_lbl_val] = new_lbl_val
                    elif key_name == "sign":
                        new_lbl_val = 12
                        # print("label2-3_new", new_lbl_val)
                        lbl_tmp[lbl == old_lbl_val] = new_lbl_val
                    elif key_name == "pedestrian":
                        new_lbl_val = 13
                        # print("label2-3_new", new_lbl_val)
                        lbl_tmp[lbl == old_lbl_val] = new_lbl_val
                    elif key_name == "motorcycle":
                        new_lbl_val = 14
                        # print("label2-3_new", new_lbl_val)
                        lbl_tmp[lbl == old_lbl_val] = new_lbl_val
                    elif key_name == "water":
                        new_lbl_val = 15
                        # print("label2-3_new", new_lbl_val)
                        lbl_tmp[lbl == old_lbl_val] = new_lbl_val
                    elif key_name == "roadblock":
                        new_lbl_val = 16
                        # print("label2-3_new", new_lbl_val)
                        lbl_tmp[lbl == old_lbl_val] = new_lbl_val
                    elif key_name == "bicycle":
                        new_lbl_val = 17
                        # print("label2-3_new", new_lbl_val)
                        lbl_tmp[lbl == old_lbl_val] = new_lbl_val

                    # print("label3", lbl)
                    # print("T2", new_lbl_val)
                print("T3")
                lbl_names_tmp = {}
                for key_name in lbl_names:
                    lbl_names_tmp[key_name] = NAME_LABEL_MAP[key_name]
                    # print("label4", lbl_names_tmp[key_name])

                # Assign the new label to lbland lbl _names dict
                lbl = np.array(lbl_tmp, dtype=np.int8)
                # for i in lbl:
                #     print("label5",i)
                lbl_names = NAME_LABEL_MAP
                # print("label5", lbl_names)
                #captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
                captions = ['%d: %s' % (l, name) for name, l in lbl_names.items()]
                #captions = []
                # print("+++++++++++++++++++++++++++++")
                # for index, word in lbl_names:
                #     content = '%d: %s'%(word, index)
                #     print("content:{}".format(content))
                #     captions.append(content)
                print("label6", captions)
                # lbl_viz = utils.draw.draw_label(lbl, img, captions)
                print("label7")
                out_dir = osp.basename(list[i]).replace('.', '_')

                out_dir = osp.join(osp.dirname(list[i]), out_dir)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if not os.path.exists(out_path + 'label_' + data_flag):
                    os.mkdir(out_path + 'label_' + data_flag)
                    os.mkdir(out_path + 'img_' + data_flag)
                    os.mkdir(out_path + 'label_viz_' + data_flag)
                label_path = osp.join(out_path + 'label_' + data_flag + "/", '{}.png'.format(filename))
                img_path = osp.join(out_path + 'img_' + data_flag + "/", '{}.png'.format(filename))
                label_viz_path = osp.join(out_path + 'label_viz_' + data_flag + "/", '{}.png'.format(filename))
                # print("label_viz_path", label_viz_path)
                # label_path_name = cv2.imdecode(np.fromfile(label_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                # img_path_name = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                # label_viz_path_name = cv2.imdecode(np.fromfile(label_viz_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                # cv2.imshow("label", lbl)
                # cv2.waitKey(0)
                a = lbl
                cv2.imwrite(label_path, a)
                # for i in a:
                #     print(i)
                # print(lbl*125)
                cv2.imwrite(img_path, img)
                # cv2.imwrite(label_viz_path, lbl_viz)

                # PIL.Image.fromarray(img).save(osp.join('/data/lane_detection/Codes-for-Lane-Detection/SCNN-Tensorflow/lane-detection-model/label_file/img/', '{}.png'.format(filename)))
                # PIL.Image.fromarray(lbl).save(osp.join('/data/lane_detection/Codes-for-Lane-Detection/SCNN-Tensorflow/lane-detection-model/label_file/label/', '{}.png'.format(filename)))
                # PIL.Image.fromarray(lbl_viz).save(osp.join('/data/Mask_RCNN/samples/shapes/9.6/json/', '{}_viz.png'.format(filename)))

                # with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
                #  for lbl_name in lbl_names:
                #    f.write(lbl_name + '\n')

                # warnings.warn('info.yaml is being replaced by label_names.txt')
                # info = dict(label_names=lbl_names)
                # with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
                # yaml.safe_dump(info, f, default_flow_style=False)

                print('Saved to: %s' % out_dir)
        except:
            print(os.path.join(json_file, list[i]), 'error!!!')


if __name__ == '__main__':
    main()
