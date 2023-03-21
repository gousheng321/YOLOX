import numpy as np
import math
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
                coord = get_ground_coord(f=20.17, dx=0.023, dy=0.023, h=2045, alpha=0.2888,
                                         u0=366.5 / 2, v0=305.8 / 2, boxes=boxes)
                if coord:
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


if __name__ == '__main__':
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # pos_file = '../assets/gt_terrace1.txt'
    # camera_file = '../assets/terrace1-c0.avi'
    # outputs_file = 'outputs'
    # ground_coord(pos_file, camera_file, outputs_file)

    import csv

    with open('outputs/gt_terrace1.csv', 'r', encoding='utf-8') as csv_gt:
        with open('outputs/gv_terrace1.csv', 'r', encoding='utf-8') as csv_gv:
            reader_gt = csv.reader(csv_gt)
            reader_gv = csv.reader(csv_gv)

            gt = np.empty((0, 9, 2))
            for row in csv_gt:
                arr_row = [eval(n) for n in eval(row)]
                gt = np.append(gt, [arr_row], axis=0)

            gv = np.empty(shape=[0, 3])
            count = 0
            for row in csv_gv:
                if len(row) > 1:
                    arr_row = eval(eval(row))
                    arr_row.append(1)
                    gv = np.append(gv, [arr_row], axis=0)
                    count += 1
                if count > 12:
                    break

    # M, resid, rank, sing = np.linalg.lstsq(gv[10:13], gt[10:13], rcond=None)
    # M = M.T
    # for i in range(0, 13):
        # print(gt[i].T)
        # print(np.round(M @ gv[i].T))
        # print(np.round(M @ gv[i].T) - gt[i].T)
        # print()

    np.set_printoptions(suppress=True)
    for i in range(0, 13):
        print(math.sqrt((gt[i+4][0][0] - 6466) ** 2 + (gt[i+4][0][1] + 1562) ** 2))
        print(math.sqrt(gv[i][0] ** 2 + gv[i][1] ** 2))
        print()

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
                coord = get_ground_coord(f=20.16, dx=0.023, dy=0.023, h=2045, alpha=0.2888, u0=360/2, v0=288/2,
                                         boxes=[[pix[count][0]/2, pix[count][1]/2, pix[count][0]/2, pix[count][1]/2]])
                ground = np.append(ground, np.concatenate((coord, np.array([[2045]])), axis=1), axis=0)
                count += 1

    dis_truth = np.round(np.linalg.norm(cam, axis=1))
    dis_cal = np.round(np.linalg.norm(ground, axis=1))

    for i in range(len(dis_truth)):
        print(dis_truth[i])
        print(dis_cal[i])
        print()
