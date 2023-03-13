import numpy as np
import cv2
from ranging import create_predictor, get_boxes, get_ground_coord


def grid_to_tv(
        pos,
        grid_width=30,
        grid_height=44,
        tv_origin_x=-500,
        tv_origin_y=-1500,
        tv_width=7500,
        tv_height=11000):
    tv_x = ((pos % grid_width) + 0.5) * (tv_width / grid_width) + tv_origin_x
    tv_y = ((pos / grid_width) + 0.5) * (tv_height / grid_height) + tv_origin_y

    return [tv_x, tv_y, 1]


if __name__ == '__main__':
    pos = np.loadtxt('../assets/gt_terrace1.txt', skiprows=2)
    truth = np.array([])
    for i in range(6, 10):
        truth = np.append(truth, grid_to_tv(pos[i * 25][0]))

    test_truth = np.array([])
    for i in range(10, 14):
        test_truth = np.append(test_truth, grid_to_tv(pos[i * 25][0]))

    # 读取视频
    cap = cv2.VideoCapture('../assets/terrace1-c0.avi')

    # 生成预测器
    predictor = create_predictor(exp_name='yolox-s', conf=0.4, nms=0.6, device='gpu')

    count = 1
    coord = np.array([])
    test_coord = np.array([])
    while True:
        # 读取一帧图像
        ret, frame = cap.read()

        # 获取人检测框与置信度集合
        boxes, scores = get_boxes(predictor, frame)
        cls_ids = [0 for n in range(len(boxes))]

        if 225 >= count >= 150 and count % 25 == 0:
            temp = get_ground_coord(f=19.9, dx=0.023, dy=0.023, h=2000, alpha=0.22,
                                    u0=355 / 2, v0=241 / 2, boxes=boxes)[0]
            temp.append(1)
            coord = np.append(coord, temp)

        if 325 >= count >= 250 and count % 25 == 0:
            temp = get_ground_coord(f=19.9, dx=0.023, dy=0.023, h=2000, alpha=0.22,
                                    u0=355 / 2, v0=241 / 2, boxes=boxes)[0]
            temp.append(1)
            test_coord = np.append(test_coord, temp)

        if count > 325:
            break
        count += 1

    truth = truth.reshape(4, 3)
    test_truth = test_truth.reshape(4, 3)

    coord = coord.reshape(4, 3)
    test_coord = test_coord.reshape(4, 3)

    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # 求解变换矩阵 H
    H = np.linalg.solve(coord.T.dot(coord), coord.T.dot(truth))

    print(truth)
    print(coord)
    # print(np.dot(coord, H))
    # print(H)
    print(test_truth)
    print(test_truth)
    # print(np.dot(test_coord, H))
