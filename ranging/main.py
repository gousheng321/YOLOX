import math
import time
import cv2
import torch

from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from tools.demo import Predictor
from yolox.utils import vis


def create_predictor(exp_name, conf, nms, tsize=640, device='cpu'):
    exp = get_exp(exp_name=exp_name)

    exp.test_conf = conf
    exp.nmsthre = nms
    exp.test_size = (tsize, tsize)

    model = exp.get_model()

    if device == 'gpu':
        model.cuda()

    model.eval()

    ckpt = torch.load('../weight/yolox_s.pth', map_location='cpu')
    model.load_state_dict(ckpt["model"])

    predictor = Predictor(
        model, exp, COCO_CLASSES, None, None,
        device, False, False,
    )

    return predictor


def get_boxes(predictor, img):
    outputs, img_info = predictor.inference(img)

    if outputs[0] is None:
        return img
    outputs = outputs[0].cpu()

    boxes = outputs[:, 0:4]
    boxes /= img_info['ratio']  # preprocessing: resize

    cls_ids = outputs[:, 6]
    scores = outputs[:, 4] * outputs[:, 5]

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


def get_ground_coord(f, dx, dy, h, alpha, u0, v0, boxes):
    import math

    ground_coord = []
    for i in range(len(boxes)):
        box = boxes[i]
        u = (box[0] + box[2]) / 2
        v = max(box[1], box[3])

        hi = abs(v - v0)
        wi = abs(u - u0)

        beta = math.atan(hi / (f / dy))

        y = h / math.tan(alpha + beta) if v - v0 >= 0 else h / math.tan(alpha - beta)
        x = wi * dx * math.sqrt(h**2 + y**2) / math.sqrt((hi * dy)**2 + f**2)
        x = x if u - u0 >= 0 else -x

        ground_coord.append([x, y])

    return ground_coord


if __name__ == "__main__":
    cap = cv2.VideoCapture('../assets/terrace1-c0.avi')

    fps_start_time = time.time()  # 开始计时
    fps = 0  # 初始化帧数计数器

    predictor = create_predictor(exp_name='yolox-s', conf=0.25, nms=0.45, device='gpu')

    while True:
        # 从摄像头中读取一帧图像
        ret, frame = cap.read()

        boxes, scores = get_boxes(predictor, frame)
        cls_ids = [0 for n in range(len(boxes))]

        frame = vis(img=frame, boxes=boxes, scores=scores,
                    cls_ids=cls_ids, conf=predictor.confthre, class_names=predictor.cls_names)

        coord = get_ground_coord(15, 0.0208, 0.0185, 1050, 10 * math.pi / 180,
                                 frame.shape[1] / 2, frame.shape[0] / 2, boxes)

        for i in range(len(boxes)):
            cv2.putText(frame, '({:.1f}, {:.1f})'.format(coord[i][0] / 1000, coord[i][1] / 1000),
                        (boxes[i][0], boxes[i][3]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        fps_end_time = time.time()  # 获取当前时间戳
        time_diff = fps_end_time - fps_start_time  # 计算时间差
        if time_diff > 0:
            fps = int(1 / time_diff)  # 计算FPS

        # 在视频帧上添加FPS文本
        cv2.putText(frame, 'FPS: ' + str(fps), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

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


