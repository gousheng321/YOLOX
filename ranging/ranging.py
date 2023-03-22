import os.path
import time
import cv2
import torch
import numpy as np
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from demo import Predictor


def create_predictor(exp_name, conf, nms, tsize=640, device='cpu'):
    """
    生成预测器
    @param exp_name: 实验环境名称，如 'yolo-s'
    @param conf: 置信度阈值
    @param nms: 非极大值抑制阈值
    @param tsize: 图片缩放指定大小
    @param device: 运行模型的设备，cpu或gpu
    @return: 预测器
    """
    exp = get_exp(exp_name=exp_name)

    exp.test_conf = conf
    exp.nmsthre = nms
    exp.test_size = (tsize, tsize)

    model = exp.get_model()

    if device == 'gpu':
        model.cuda()

    model.eval()

    from pathlib import Path
    model_path = Path(os.path.dirname(__file__), 'weight', '{}.pth'.format(exp_name.replace('-', '_')))

    ckpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt["model"])

    predictor = Predictor(
        model, exp, COCO_CLASSES, None, None,
        device, False, False,
    )
    return predictor


def get_boxes(predictor, img):
    """
    获取人的检测框与置信度
    @param predictor: 预测器
    @param img: 待检测图片
    @return: 人的检测框，置信度
    """
    # 预测图像中的目标
    outputs, img_info = predictor.inference(img)

    # 若图像中没有目标则返回
    if outputs[0] is None:
        return [], []
    outputs = outputs[0].cpu()

    # 所有目标的检测框
    boxes = outputs[:, 0:4]
    # 按照原图像比率调整检测框大小
    boxes /= img_info['ratio']

    # 检测类别与置信度
    cls_ids = outputs[:, 6]
    scores = outputs[:, 4] * outputs[:, 5]

    # 过滤除人以外的检测框与置信度
    man_boxes = []
    man_scores = []
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = 0
        if int(cls_ids[i]) == cls_id:
            score = scores[i]
            man_scores.append(float(score))
            if score < predictor.confthre:
                continue
            man_boxes.append([int(n) for n in box])

    return np.array(man_boxes), np.array(man_scores)


def eulerAnglesToRotationMatrix(theta, trans):
    import numpy as np
    import math
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    # 旋转矩阵
    R = np.dot(R_z, np.dot(R_y, R_x))

    # 测距俯仰角
    Zw = np.array([0, 0, 1])
    Zc = R @ Zw
    pitch = math.acos(np.dot(Zw, Zc)) - math.pi / 2

    # 相机世界坐标
    inverse_R = np.linalg.inv(R)
    Pw0 = -inverse_R @ trans

    return R, pitch, Pw0


def get_ground_coord(f, dx, dy, h, alpha, u0, v0, boxes):
    """
    获取所有检测框世界坐标
    :param f: 焦距
    :param dx: 像素横轴尺寸
    :param dy: 像素纵轴尺寸
    :param h: 相机安装高度
    :param alpha: 相机俯仰角
    :param u0: 图片中心横坐标
    :param v0: 图片中心纵坐标
    :param boxes: 人的检测框集合
    :return: 人的世界坐标集合
    """
    import math

    ground_coord = []  # 目标所在地平面世界坐标集合
    for i in range(len(boxes)):
        box = boxes[i]
        # 获取检测框底边中心坐标
        u = (box[0] + box[2]) / 2
        v = max(box[1], box[3])

        # 落脚点成像宽高
        hi = abs(v - v0) * 2
        wi = (u - u0) * 2

        # 成像点射线与光轴在y轴的夹角
        beta = math.atan(hi * dy / f)

        # 计算落脚点地面坐标
        y = (h / math.tan(alpha + beta)) if (v - v0 >= 0) else (h / math.tan(alpha - beta))
        x = wi * dx * math.sqrt(h ** 2 + y ** 2) / math.sqrt((hi * dy) ** 2 + f ** 2)

        ground_coord.append([x, y])

    return np.round(ground_coord)


def rule_1meter(ground_coord):
    import numpy as np

    rules = []
    for i in range(len(ground_coord)):
        for j in range(len(ground_coord)):
            if i != j:
                dis = np.linalg.norm(ground_coord[i] - ground_coord[j])
                if dis < 1000:
                    rules.append(False)
                    break
        if len(rules) < i + 1:
            rules.append(True)

    return rules


def visualize(img, boxes, show_dis=False, show_coord=False, activate_rule=False):
    if show_dis and len(coord) > 0:
        dis = np.linalg.norm(coord, axis=1)
        for i in range(len(boxes)):
            text = '{:.2f}m'.format(dis[i] / 1000)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.4
            thickness = 1
            size, baseLine = cv2.getTextSize(text, font, fontScale, thickness)
            x, y = (boxes[i][0] + boxes[i][2]) / 2 - size[0] / 2, boxes[i][3] - baseLine
            cv2.putText(img, text, (int(x), int(y)), font, fontScale, (255, 255, 255), thickness)

    if show_coord:
        for i in range(len(boxes)):
            text = '({:.1f}, {:.1f})'.format(coord[i][0] / 1000, coord[i][1] / 1000)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.3
            thickness = 1
            size, baseLine = cv2.getTextSize(text, font, fontScale, thickness)
            x, y = (boxes[i][0] + boxes[i][2]) / 2 - size[0] / 2, boxes[i][3] + size[1] + baseLine
            cv2.putText(img, text, (int(x), int(y)), font, fontScale, (255, 255, 255), thickness)

    if activate_rule:
        # 计算坐标是否遵守一米距规则
        rules = rule_1meter(coord)
        # 过滤合规检测框
        boxes = [boxes[n] for n in range(len(rules)) if not rules[n]]

    for i in range(len(boxes)):
        box = boxes[i]
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (33, 26, 213), 1)

    return img


if __name__ == "__main__":
    import logging

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    # 读取视频
    cap = cv2.VideoCapture('../assets/terrace1-c0.avi')

    fps_start_time = time.time()  # 开始计时
    fps = 0  # 初始化帧数计数器

    # 旋转矩阵，俯仰角，相机世界坐标
    R, pitch, Pw0 = eulerAnglesToRotationMatrix([1.9007833770e+00, 4.9730769727e-01, 1.8415452559e-01],
                                                np.array([-4.8441913843e+03, 5.5109448682e+02, 4.9667438357e+03]))

    # 生成预测器
    predictor = create_predictor(exp_name='yolox-s', conf=0.5, nms=0.4, device='gpu')

    while True:
        # 读取一帧图像
        ret, frame = cap.read()
        if not ret:
            logging.info('End of video')
            break

        # 获取人检测框与置信度集合
        boxes, scores = get_boxes(predictor, frame)
        # 过滤分数低于置信度的检测框
        boxes = [boxes[n] for n in range(len(boxes)) if scores[n] >= predictor.confthre]

        # 计算目标世界坐标集合
        coord = get_ground_coord(f=20.17, dx=0.023, dy=0.023, h=2045, alpha=pitch,
                                 u0=366 / 2, v0=305 / 2, boxes=boxes)

        # 绘制检测框
        frame = visualize(img=frame, boxes=boxes, show_dis=True, show_coord=True, activate_rule=True)

        fps_end_time = time.time()  # 获取当前时间戳
        time_diff = fps_end_time - fps_start_time  # 计算时间差
        if time_diff > 0:
            fps = int(1 / time_diff)  # 计算FPS

        # 在视频帧上添加FPS文本
        cv2.putText(frame, 'FPS: ' + str(fps), (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        # 缩放图像
        resized_img = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2))
        # 显示图像
        cv2.imshow('Detect', resized_img)
        # 更新计时器
        fps_start_time = fps_end_time
        # 等待按键事件，如果按下q键，则退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头资源
    cap.release()
    # 关闭所有窗口
    cv2.destroyAllWindows()
