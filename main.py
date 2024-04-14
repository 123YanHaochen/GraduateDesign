
import time
from hardware.connection import *
import cv2
import numpy as np
import argparse
import onnxruntime as ort
import math

class yolov5_lite():
    def __init__(self, model_pb_path, label_path, confThreshold=0.5, nmsThreshold=0.5, objThreshold=0.5):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.net = ort.InferenceSession(model_pb_path, so)
        self.classes = list(map(lambda x: x.strip(), open(label_path, 'r').readlines()))
        self.num_classes = len(self.classes)
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.no = self.num_classes + 5
        self.grid = [np.zeros(1)] * self.nl
        self.stride = np.array([8., 16., 32.])
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, -1, 2)

        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold
        self.input_shape = (self.net.get_inputs()[0].shape[2], self.net.get_inputs()[0].shape[3])
        
    def resize_image(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_shape[0], self.input_shape[1]
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_shape[0], int(self.input_shape[1] / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_shape[1] - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.input_shape[1] - neww - left, cv2.BORDER_CONSTANT,
                                         value=0)  # add border
            else:
                newh, neww = int(self.input_shape[0] * hw_scale), self.input_shape[1]
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_shape[0] - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.input_shape[0] - newh - top, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(srcimg, self.input_shape, interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left
    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def postprocess(self, frame, outs, pad_hw):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        #把[-frameWidth/2, frameWidth/2] -> [0,127] 中心为 63
        #  [-frameHeight/2, frameHeight/2] -> [128, 255] 中心为191
        bias_x, bias_y = 64, 192
        newh, neww, padh, padw = pad_hw

        ratioh, ratiow = frameHeight / newh, frameWidth / neww
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for detection in outs:
            scores = detection[5:]
            classId = np.argmax(scores)

            confidence = scores[classId]
            if confidence > self.confThreshold and detection[4] > self.objThreshold:
                center_x = int((detection[0] - padw) * ratiow)
                center_y = int((detection[1] - padh) * ratioh)
                width = int(detection[2] * ratiow)
                height = int(detection[3] * ratioh)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        hasPeople = 0
        for i in indices:
            i = i[0] if isinstance(i, (tuple,list)) else i
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            #输出左上角的横纵坐标 图像的高和宽
            # O------> X轴
            # |
            # |
            # v Y轴

            if classIds[i] == 0: # 0 -> people
                if hasPeople == 1: continue
                hasPeople = 1

                # print("+++++++++++++++++++++++++++++++++++++++++")
                # print(f"class:{self.classes[classIds[i]]}" )
                # print(f"Portrait Center: x = {left + width / 2}, y = {top + height / 2}")
                # print(f"Frame Center: x = {frameWidth / 2}, y = {frameHeight / 2}")
                # print("+++++++++++++++++++++++++++++++++++++++++")

                #bias = taraget - current
                bias_x = frameWidth / 2 - (left + width / 2)
                bias_y = frameHeight / 2 - (top + height / 2)
                #对x进行映射
                if bias_x < -320: bias_x = -320
                elif bias_x > 320: bias_x = 320
                bias_x = (bias_x + 320) / 5
                #对y进行映射
                if bias_y < -(frameHeight / 2): bias_y =  -(frameHeight // 2)
                elif bias_y > frameHeight / 2: bias_y = frameHeight // 2
                bias_y = (bias_y + frameHeight // 2) / (frameHeight // 128) + 128


            #如果bias_x 没有赋值

            frame = self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
        #print(f"bias_x = {bias_x}, bias_y = {bias_y}")
        return frame, bias_x, bias_y
    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=4)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        return frame
    def detect(self, srcimg):
        img, newh, neww, top, left = self.resize_image(srcimg)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

        outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})[0].squeeze(axis=0)
        row_ind = 0
        for i in range(self.nl):
            h, w = int(self.input_shape[0] / self.stride[i]), int(self.input_shape[1] / self.stride[i])
            length = int(self.na * h * w)
            if self.grid[i].shape[2:4] != (h, w):
                self.grid[i] = self._make_grid(w, h)

            outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
                self.grid[i], (self.na, 1))) * int(self.stride[i])
            outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
                self.anchor_grid[i], h * w, axis=0)
            row_ind += length
        srcimg, bias_x, bias_y = self.postprocess(srcimg, outs, (newh, neww, top, left))
#         cv2.imwrite('result.jpg', srcimg)
        return srcimg, bias_x, bias_y

if __name__=='__main__':

    #打开串口
    serial = serial.Serial('COM9', 9600, timeout=0.5)
    if serial.isOpen():
        print("open success")
    else:
        print("open failed")

    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='imgs/street.png', help="image path")
    parser.add_argument('--modelpath', type=str, default='onnxmodel/v5lite-c.onnx', help="onnx filepath")
    parser.add_argument('--classfile', type=str, default='imgs/coco.names', help="classname filepath")
    parser.add_argument('--confThreshold', default=0.5, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.6, type=float, help='nms iou thresh')
    args = parser.parse_args()


    net = yolov5_lite(args.modelpath, args.classfile, confThreshold=args.confThreshold, nmsThreshold=args.nmsThreshold)

    # 打开摄像头
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # 0 表示默认摄像头，如果有多个摄像头，可以逐一尝试
    #cap = cv2.VideoCapture("http://admin:admin@192.168.43.1:8081/")

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    winName = 'Deep learning object detection in onnxruntime'
    start_time = time.time()
    frame_count = 0
    # 循环读取帧并进行检测
    while True:
        # 读取摄像头帧
        ret, frame = cap.read()

        if not ret:
            print("无法接收摄像头数据")
            break
        # 对帧进行检测
        frame, bias_x, bias_y = net.detect(frame)
        #传送数据
        send(serial, int(bias_x).to_bytes(3, 'big'))  # big表示最高位在前
        sleep(0.1)
        send(serial, int(bias_y).to_bytes(3, 'big'))  # big表示最高位在前

        # 计算帧率
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time

        # 在右上角显示帧率
        cv2.putText(frame, f'FPS: {round(fps, 2)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)



        # 显示检测结果
        cv2.imshow(winName, frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头资源并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()



















    # srcimg = cv2.imread(args.imgpath)
    # net = yolov5_lite(args.modelpath, args.classfile, confThreshold=args.confThreshold, nmsThreshold=args.nmsThreshold)
    # srcimg = net.detect(srcimg)
    #
    # winName = 'Deep learning object detection in onnxruntime'
    # cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    # cv2.imshow(winName, srcimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()