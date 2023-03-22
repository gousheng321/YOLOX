import numpy as np
import math
import cv2
from ranging import create_predictor, get_boxes, get_ground_coord, rule_1meter


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


def calc_ground_coord(pos_file, camera_file, save_path):
    import csv

    pos = np.loadtxt(pos_file, skiprows=2)
    cap = cv2.VideoCapture(camera_file)
    predictor = create_predictor(exp_name='yolox-s', conf=0.5, nms=0.4, device='gpu')

    with open('{}/gt_terrace1.csv'.format(save_path), 'w', encoding='utf-8', newline='') as csv_gt:
        with open('{}/gc_terrace1.csv'.format(save_path), 'w', encoding='utf-8', newline='') as csv_gv:
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
                coord = get_ground_coord(f=20.17, dx=0.023, dy=0.023, h=2045, alpha=0.2888,
                                         u0=366.5 / 2, v0=305.8 / 2, boxes=boxes)
                if coord.all():
                    coord = np.round(coord).tolist()
                    writer_gv.writerow([n for n in coord])
                else:
                    writer_gv.writerow('')


def camera_coord(world_coord):
    R = np.array([[0.86400969, 0.50302581, 0.02126701], [0.16093467, -0.23590786, -0.95835667],
                  [-0.47706109, 0.83145205, -0.28478097]])
    T = np.array([[-4.8441913843e+03], [5.5109448682e+02], [4.9667438357e+03]])

    M = np.concatenate((R, T), axis=1)
    M = np.concatenate((M, np.array([[0, 0, 0, 1]])), axis=0)

    return M @ world_coord


def pixel_coord(camera_coord):
    M = np.array([[20.16192 / 2.3000000000e-02, 0, 366.514507, 0],
                  [0, 20.16192 / 2.3000000000e-02, 305.832552, 0],
                  [0, 0, 1, 0]])

    return M @ camera_coord / camera_coord[2]


def test_ground_coord(gt):
    wor = np.empty(shape=[0, 4])
    cam = np.empty(shape=[0, 4])
    pix = np.empty(shape=[0, 3])
    ground = np.empty(shape=[0, 3])

    count = 0
    for i in range(len(gt)):
        for j in range(len(gt[i])):
            if sum(gt[i][j]) % 250 == 0:
                wor = np.append(wor, [np.array([gt[i][j][0], gt[i][j][1], 0, 1])], axis=0)
                cam = np.append(cam, [camera_coord(wor[count])], axis=0)
                pix = np.append(pix, [pixel_coord(cam[count])], axis=0)
                coord = get_ground_coord(f=20.16, dx=0.023, dy=0.023, h=2045, alpha=0.2888, u0=306 / 2, v0=305 / 2,
                                         boxes=[[pix[count][0] / 2, pix[count][1] / 2, pix[count][0] / 2,
                                                 pix[count][1] / 2]])
                ground = np.append(ground, np.concatenate((coord, np.array([[2045]])), axis=1), axis=0)
                count += 1

    coord_arr = np.column_stack((wor, cam, pix, ground))
    np.savetxt("outputs/coord.csv", coord_arr, delimiter=",", fmt="%d",
               header="{},{},{},{}".format('世界坐标', '相机坐标', '像素坐标', '地面坐标'), comments="")

    dis_truth = np.round(np.linalg.norm(cam, axis=1))
    dis_cal = np.round(np.linalg.norm(ground, axis=1))
    err = np.abs(dis_truth - dis_cal)

    dis_arr = np.column_stack((dis_truth, dis_cal, err))
    np.savetxt("outputs/dis.csv", dis_arr, delimiter=",", fmt="%d",
               header="{},{},{}".format('距离真值', '距离计算值', '绝对误差'), comments="")

    return wor, cam, pix, ground, dis_truth, dis_cal


def test_ground_detect(gt, gc):
    set_dis_calc = np.array([])
    set_dis_truth = np.array([])
    for i, gc_v in enumerate(gc):
        if len(gc_v) <= 1:
            continue
        if type(eval(gc_v)) == str:
            calc = [eval(eval(gc_v))]
        else:
            calc = [eval(n) for n in eval(gc[i])]
        for j, val in enumerate(calc):
            dis_calc = round(np.linalg.norm(val))
            for k, gt_v in enumerate(gt[i]):
                if sum(gt_v) % 250 != 0:
                    continue
                dis_gt = round(np.linalg.norm(gt_v - np.array([6466, -1562])))
                if abs(dis_gt - dis_calc) < 250:
                    set_dis_calc = np.append(set_dis_calc, dis_calc)
                    set_dis_truth = np.append(set_dis_truth, dis_gt)
                    break

    dis_arr = np.column_stack((set_dis_truth, set_dis_calc, np.abs(set_dis_truth - set_dis_calc)))
    np.savetxt("outputs/dis_detect.csv", dis_arr, delimiter=",", fmt="%d",
               header="{},{},{}".format('距离真值', '距离计算值(采用检测模块)', '绝对误差'), comments="")

    return set_dis_truth, set_dis_calc


def test_rule_1meter(gt):

    def calc_min_dis(arr):
        dis_set = np.array([])
        min_dis = np.inf
        for i, coord in enumerate(arr):
            for j, _coord in enumerate(arr):
                if i == j:
                    continue
                dis = np.linalg.norm(coord - _coord)
                if dis < min_dis:
                    min_dis = dis
            dis_set = np.append(dis_set, min_dis)
            min_dis = np.inf
        return np.round(dis_set)

    import csv

    with open('outputs/test_rule.csv', 'w+', encoding='utf-8', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        count = 0
        length = 1
        for i, gt_val in enumerate(gt):
            fltr = np.all(gt_val != [-1, -1], axis=1)
            gt_val = gt_val[fltr]
            if len(gt_val) == length:
                csv_writer.writerow(gt_val)
                csv_writer.writerow(rule_1meter(gt_val))
                csv_writer.writerow(calc_min_dis(gt_val))
                count += 1
                if count == 5:
                    count = 0
                    length += 1
                    continue


if __name__ == '__main__':
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # pos_file = '../assets/gt_terrace1.txt'
    # camera_file = '../assets/terrace1-c0.avi'
    # outputs_file = 'outputs'
    # calc_ground_coord(pos_file, camera_file, outputs_file)

    with open('outputs/gt_terrace1.csv', 'r', encoding='utf-8') as csv_gt:
        with open('outputs/gc_terrace1.csv', 'r', encoding='utf-8') as csv_gv:
            gt = np.empty((0, 9, 2))
            for row in csv_gt:
                arr_row = [eval(n) for n in eval(row)]
                gt = np.append(gt, [arr_row], axis=0)

            gc = np.array([])
            for row in csv_gv:
                gc = np.append(gc, row)

    # test_ground_coord(gt)
    # test_ground_detect(gt, gc)

    test_rule_1meter(gt)

    # M, resid, rank, sing = np.linalg.lstsq(gv[10:13], gt[10:13], rcond=None)
    # M = M.T
    # for i in range(0, 13):
    #     print(gt[i].T)
    #     print(np.round(M @ gv[i].T))
    #     print(np.round(M @ gv[i].T) - gt[i].T)
    #     print()
