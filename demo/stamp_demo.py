# encoding: utf-8
"""
@author: Yuxian Meng 
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: stamp_demo
@time: 2019/4/26 11:00

    这一行开始写关于本文件的说明与解释
"""

import os
from maskrcnn_benchmark.config import cfg
from demo.predictor import StampDemo

config_file = "../configs/fcos/fcos_R_50_FPN_1x.yaml"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda:0",
                     "MODEL.WEIGHT", "/home/mengyuxian/FCOS/training_dir/fcos_R_50_FPN_1x/model_final.pth",
                     ])

stamp_demo = StampDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)


if __name__ == '__main__':
    # load image and then run prediction
    import cv2 as cv
    total_t = 0
    src_dir = "/data/nfsdata/data/yuxian/datasets/stamps_20190416/images/"
    tgt_dir = "/data/nfsdata/data/yuxian/datasets/stamps_predict"
    for img_path in os.listdir(src_dir)[:]:
        print(img_path)
        if img_path.endswith("png"):
            image = cv.imread(os.path.join(src_dir, img_path))
            for _ in range(1):
                predictions = stamp_demo.run_on_opencv_image(image)
                cv.imwrite(os.path.join(tgt_dir, img_path), predictions)

    # img_path = "/data/nfsdata/data/yuxian/datasets/yuxian_test/legacy.jpg"
    # tg_path = "/data/nfsdata/data/yuxian/datasets/yuxian_test/legacy_output.jpg"
    # image = cv.imread(img_path)
    # for i in range(100):
    #     predictions = stamp_demo.run_on_opencv_image(image)
        # cv.imwrite(os.path.join(tgt_dir, img_path), predictions)
