import os.path
import time
import cv2
import numpy as np
import torch

from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from demo import Predictor
from yolox.utils import vis


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

    return man_boxes, man_scores


def get_eulerAngles(rx, ry, rz):
    import numpy as np
    import math
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    r_vec = np.array([rx, ry, rz])
    R, _ = cv2.Rodrigues(r_vec)

    def isRotationMatrix(R):
        # 得到该矩阵的转置
        Rt = np.transpose(R)
        # 旋转矩阵的一个性质是，相乘后为单位阵
        shouldBeIdentity = np.dot(Rt, R)
        # 构建一个三维单位阵
        I = np.identity(3, dtype=R.dtype)
        # 将单位阵和旋转矩阵相乘后的值做差
        n = np.linalg.norm(I - shouldBeIdentity)
        # 如果小于一个极小值，则表示该矩阵为旋转矩阵
        return n < 1e-6
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    a = np.array([0, 0, 1])
    b = np.dot(R, a)
    c = np.dot(a, b)

    _euler_angles = np.array([x, y, z])

    print(f'Euler: {np.rad2deg(_euler_angles)}')

    return _euler_angles


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

    if len(boxes) == 0:
        return

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
        # vh = f / dy * math.tan(alpha)
        # y = 1 / (math.cos(alpha) ** 2) * f / dy * h / (hi + vh) - h * math.tan(alpha) if hi + vh >= 0 else -1
        # x = wi * h / (hi + vh) * 1 / math.cos(alpha)

        y = h / math.tan(alpha + beta) if v - v0 >= 0 else h / math.tan(alpha - beta)
        x = wi * dx * math.sqrt(h ** 2 + y ** 2) / math.sqrt((hi * dy) ** 2 + f ** 2)

        ground_coord.append([x, y])

    return ground_coord


if __name__ == "__main__":
    import logging

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    # 读取视频
    cap = cv2.VideoCapture('../assets/terrace1-c2.avi')

    fps_start_time = time.time()  # 开始计时
    fps = 0  # 初始化帧数计数器

    # 生成预测器
    predictor = create_predictor(exp_name='yolox-s', conf=0.4, nms=0.6, device='gpu')

    while True:
        # 读取一帧图像
        ret, frame = cap.read()
        if not ret:
            logging.info('End of video')
            break

        # 获取人检测框与置信度集合
        boxes, scores = get_boxes(predictor, frame)
        cls_ids = [0 for n in range(len(boxes))]

        # 绘制检测框
        frame = vis(img=frame, boxes=boxes, scores=scores,
                    cls_ids=cls_ids, conf=predictor.confthre, class_names=predictor.cls_names)

        euler_angles = get_eulerAngles(-1.83, 0.377, 3.02)

        # 计算目标世界坐标集合
        coord = get_ground_coord(f=19.9, dx=0.023, dy=0.023, h=2000, alpha=0.25,
                                 u0=355 / 2, v0=241 / 2, boxes=boxes)

        import math

        # 添加坐标文本
        for i in range(len(boxes)):
            cv2.putText(frame, '({:.1f} , {:.1f})'.format(coord[i][0] / 1000, coord[i][1] / 1000),
                        (boxes[i][0], boxes[i][3]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        fps_end_time = time.time()  # 获取当前时间戳
        time_diff = fps_end_time - fps_start_time  # 计算时间差
        if time_diff > 0:
            fps = int(1 / time_diff)  # 计算FPS

        # 在视频帧上添加FPS文本
        cv2.putText(frame, 'FPS: ' + str(fps), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
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
