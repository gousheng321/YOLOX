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
    if pos < 0:
        return [-1, -1]
    tv_x = int((pos % grid_width) + 0.5) * (tv_width / grid_width) + tv_origin_x
    tv_y = int((pos / grid_width) + 0.5) * (tv_height / grid_height) + tv_origin_y

    return np.array([tv_x, tv_y])


def ground_coord(pos_file, camera_file, save_path):
    import csv

    pos = np.loadtxt(pos_file, skiprows=2)
    cap = cv2.VideoCapture(camera_file)
    predictor = create_predictor(exp_name='yolox-s', conf=0.4, nms=0.6, device='gpu')

    with open('{}/gt_terrace1.csv'.format(save_path), 'w', encoding='utf-8', newline='') as csv_gt:
        with open('{}/gv_terrace1.csv'.format(save_path), 'w', encoding='utf-8', newline='') as csv_gv:
            writer_gt = csv.writer(csv_gt, delimiter=',')
            writer_gv = csv.writer(csv_gv)

            for row in range(len(pos)):
                ret, frame = cap.read()
                if not ret:
                    break
                if row % 25 != 0 or row == 0:
                    continue

                truth = np.empty(shape=[0, 2])
                for col in range(len(pos[row])):
                    rv = grid_to_tv(pos[row][col])
                    truth = np.append(truth, [rv], axis=0).tolist()
                writer_gt.writerow([n for n in truth])

                boxes, scores = get_boxes(predictor, frame)
                coord = get_ground_coord(f=19.9, dx=0.023, dy=0.023, h=2000, alpha=0.28,
                                         u0=355 / 2, v0=241 / 2, boxes=boxes)
                if coord:
                    coord = np.round(coord).tolist()
                    writer_gv.writerow([n for n in coord])
                else:
                    writer_gv.writerow('')


if __name__ == '__main__':
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # pos_file = '../assets/gt_terrace1.txt'
    # camera_file = '../assets/terrace1-c2.avi'
    # outputs_file = 'outputs'
    # ground_coord(pos_file, camera_file, outputs_file)

    import csv
    with open('outputs/gt_terrace1.csv', 'r', encoding='utf-8') as csv_gt:
        with open('outputs/gv_terrace1.csv', 'U', encoding='utf-8') as csv_gv:
            reader_gt = csv.reader(csv_gt)
            reader_gv = csv.reader(csv_gv)

            gt = np.empty(shape=[0, 2])
            count = 0

            for row in csv_gt:
                arr_row = [eval(n) for n in eval(row)]
                if arr_row[0][0] != -1:
                    gt = np.append(gt, [arr_row[0]], axis=0)
                    count += 1
                if count == 10:
                    break
            print(gt)

            gv = np.empty(shape=[0, 2])
            for row in csv_gv:
                if len(row) > 1:
                    a=eval(row)
                    b=eval(a)
                    arr_row = [eval(n) for n in eval(row)]
                    gt = np.append(gt, arr_row[0], axis=0)
            print(gv)



    # 求解变换矩阵 H
    # H = np.linalg.solve(coord.T.dot(coord), coord.T.dot(truth))


