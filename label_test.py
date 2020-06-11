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
    "bicycle": 17
}

# NAME_LABEL_MAP = {
#     '_background_': 0,
#     "road": 1,
#     "lane": 2,
#     "sign": 3,
#     "pavement": 4,
#     "pedestrain": 5,
#     "grass": 6,
#     "building": 7,
#     "lamp": 8,
#     "person": 9,
#     "car": 10
# }
out_path = 'C:\\Users\\Nicolas Acci\\Desktop\\ist\\all/'

def main(out_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', default='label_file/road_label')
    parser.add_argument('-o', '--out', default=None)
    args = parser.parse_args()


    json_file = out_path
    print("json_file", json_file)
    list = os.listdir(json_file)
    print("list",list)

    for i in range(0, len(list)):
        try:
            path = os.path.join(json_file, list[i])
            print(path)
            filename = list[i][:-5]  # .json
            if os.path.isfile(path):
                data = json.load(open(path))
                img = utils.image.img_b64_to_arr(data['imageData'])
                lbl, lbl_names = utils.shape.labelme_shapes_to_label(img.shape,
                                                                     data['shapes'])  # labelme_shapes_to_label
                print("label", lbl.shape, lbl_names)
                # modify labels according to NAME_LABEL_MAP
                lbl_tmp = copy.copy(lbl)
                for key_name in lbl_names:
                    old_lbl_val = lbl_names[key_name]
                    new_lbl_val = NAME_LABEL_MAP[key_name]
                    # 设定规定值的标签图像
                    lbl_tmp[lbl == old_lbl_val] = new_lbl_val
                lbl_names_tmp = {}
                for key_name in lbl_names:
                    lbl_names_tmp[key_name] = NAME_LABEL_MAP[key_name]

                # Assign the new label to lbl and lbl_names dict
                lbl = np.array(lbl_tmp, dtype=np.int8)
                lbl_names = NAME_LABEL_MAP

                captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
                lbl_viz = utils.draw.draw_label(lbl, img, captions)
                out_dir = osp.basename(list[i]).replace('.', '_')
                out_dir = osp.join(osp.dirname(list[i]), out_dir)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if not os.path.exists(out_path + 'label'):
                    os.mkdir(out_path + 'label')
                    os.mkdir(out_path + 'img')
                    os.mkdir(out_path + 'label_viz')
                label_path = osp.join(out_path + 'label' + "/", '{}.png'.format(filename))
                img_path = osp.join(out_path + 'img' + "/", '{}.png'.format(filename))
                label_viz_path = osp.join(out_path + 'label_viz' + "/", '{}.png'.format(filename))
                # print("label_viz_path", label_viz_path)
                # label_path_name = cv2.imdecode(np.fromfile(label_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                # img_path_name = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                # label_viz_path_name = cv2.imdecode(np.fromfile(label_viz_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                # cv2.imshow("label", lbl)
                # cv2.waitKey(0)
                # cv2.imwrite(label_path, lbl)
                cv2.imwrite(img_path, img)
                print(img_path)
                cv2.imwrite(label_viz_path, lbl_viz)

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
    main(out_path)
