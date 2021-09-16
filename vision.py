import onnxruntime as rt
import numpy as np
from PIL import Image
# run this program on each RPi to send a labelled image stream
import socket
import time
from imutils.video import VideoStream
import imutils
import imagezmq
import cv2
import time


class Vision:
    def __init__(self) -> None:
        self.sender = imagezmq.ImageSender(
            connect_to='tcp://192.168.28.49:5555')
        self.input_shape = (416, 416)
        self.rpi_name = socket.gethostname()  # send RPi hostname with each image
        self.picam = VideoStream(
            usePiCamera=True, resolution=self.input_shape).start()
        time.sleep(2.0)  # allow camera sensor to warm up
        self.so = rt.SessionOptions()
        # so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.text_file = open("class_list.txt", "r")
        self.CLASSES = self.text_file.read().split('\n')
        self.sess = rt.InferenceSession(r'yolox_tiny_v1_1.onnx', self.so)
        self.result = "e"
        self._COLORS = np.array(
            [
                0.000, 0.447, 0.741,
                0.850, 0.325, 0.098,
                0.929, 0.694, 0.125,
                0.494, 0.184, 0.556,
                0.466, 0.674, 0.188,
                0.301, 0.745, 0.933,
                0.635, 0.078, 0.184,
                0.300, 0.300, 0.300,
                0.600, 0.600, 0.600,
                1.000, 0.000, 0.000,
                1.000, 0.500, 0.000,
                0.749, 0.749, 0.000,
                0.000, 1.000, 0.000,
                0.000, 0.000, 1.000,
                0.667, 0.000, 1.000,
                0.333, 0.333, 0.000,
                0.333, 0.667, 0.000,
                0.333, 1.000, 0.000,
                0.667, 0.333, 0.000,
                0.667, 0.667, 0.000,
                0.667, 1.000, 0.000,
                1.000, 0.333, 0.000,
                1.000, 0.667, 0.000,
                1.000, 1.000, 0.000,
                0.000, 0.333, 0.500,
                0.000, 0.667, 0.500,
                0.000, 1.000, 0.500,
                0.333, 0.000, 0.500,
                0.333, 0.333, 0.500,
                0.333, 0.667, 0.500,
                0.333, 1.000, 0.500,
                0.667, 0.000, 0.500,
                0.667, 0.333, 0.500,
                0.667, 0.667, 0.500,
                0.667, 1.000, 0.500,
                1.000, 0.000, 0.500,
                1.000, 0.333, 0.500,
                1.000, 0.667, 0.500,
                1.000, 1.000, 0.500,
                0.000, 0.333, 1.000,
                0.000, 0.667, 1.000,
                0.000, 1.000, 1.000,
                0.333, 0.000, 1.000,
                0.333, 0.333, 1.000,
                0.333, 0.667, 1.000,
                0.333, 1.000, 1.000,
                0.667, 0.000, 1.000,
                0.667, 0.333, 1.000,
                0.667, 0.667, 1.000,
                0.667, 1.000, 1.000,
                1.000, 0.000, 1.000,
                1.000, 0.333, 1.000,
                1.000, 0.667, 1.000,
                0.333, 0.000, 0.000,
                0.500, 0.000, 0.000,
                0.667, 0.000, 0.000,
                0.833, 0.000, 0.000,
                1.000, 0.000, 0.000,
                0.000, 0.167, 0.000,
                0.000, 0.333, 0.000,
                0.000, 0.500, 0.000,
                0.000, 0.667, 0.000,
                0.000, 0.833, 0.000,
                0.000, 1.000, 0.000,
                0.000, 0.000, 0.167,
                0.000, 0.000, 0.333,
                0.000, 0.000, 0.500,
                0.000, 0.000, 0.667,
                0.000, 0.000, 0.833,
                0.000, 0.000, 1.000,
                0.000, 0.000, 0.000,
                0.143, 0.143, 0.143,
                0.286, 0.286, 0.286,
                0.429, 0.429, 0.429,
                0.571, 0.571, 0.571,
                0.714, 0.714, 0.714,
                0.857, 0.857, 0.857,
                0.000, 0.447, 0.741,
                0.314, 0.717, 0.741,
                0.50, 0.5, 0
            ]
        ).astype(np.float32).reshape(-1, 3)

    def demo_postprocess(self, outputs, img_size, p6=False):

        grids = []
        expanded_strides = []

        if not p6:
            strides = [8, 16, 32]
        else:
            strides = [8, 16, 32, 64]

        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        return outputs

    def nms(self, boxes, scores, nms_thr):
        """Single class NMS implemented in Numpy."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep

    def multiclass_nms_class_agnostic(self, boxes, scores, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy. Class-agnostic version."""
        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            return None
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        keep = self.nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            dets = np.concatenate(
                [valid_boxes[keep], valid_scores[keep, None],
                    valid_cls_inds[keep, None]], 1
            )
        return dets

    def multiclass_nms(self, boxes, scores, nms_thr, score_thr, class_agnostic=True):
        """Multiclass NMS implemented in Numpy"""
        if class_agnostic:
            nms_method = self.multiclass_nms_class_agnostic
        else:
            pass
            # nms_method = self.multiclass_nms_class_aware
        return nms_method(boxes, scores, nms_thr, score_thr)

    def vis(self, img, boxes, scores, cls_ids, conf=0.5, class_names=None):

        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            if score < conf:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            color = (self._COLORS[cls_id] * 255).astype(np.uint8).tolist()
            text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
            txt_color = (0, 0, 0) if np.mean(
                self._COLORS[cls_id]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX

            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

            txt_bk_color = (self._COLORS[cls_id] * 255 *
                            0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                img,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(
                img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

        return img


   def vis(self, img, boxes, scores, cls_ids, conf=0.5, class_names=None):

        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            if score < conf:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            color = (self._COLORS[cls_id] * 255).astype(np.uint8).tolist()
            text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
            txt_color = (0, 0, 0) if np.mean(
                self._COLORS[cls_id]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX

            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

            txt_bk_color = (self._COLORS[cls_id] * 255 *
                            0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                img,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(
                img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

        return img

    def detect(self):
        image = self.picam.read()
        image = imutils.rotate(image, 180)
        img = np.float32(image)  # [:, :, [2, 1, 0]].astype('float32')
        img_data = np.transpose(img, [2, 0, 1])
        img_data = np.expand_dims(img_data, axis=0)
        input_name = self.sess.get_inputs()[0].name
        output = self.sess.run(None, {input_name: img_data})
        predictions = self.demo_postprocess(
            output[0], self.input_shape, False)[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        dets = self.multiclass_nms(
            boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:,
                                                             :4], dets[:, 4], dets[:, 5]
            image = self.vis(image, final_boxes, final_scores, final_cls_inds,
                             conf=0.3, class_names=self.CLASSES)
            # self.sender.send_image(self.rpi_name,  image)
            return True
        else:
            return False

    # prev_frame_time = 0
    # new_fram_time = 0
    # while True:  # send images as stream until Ctrl-C

    #     image = picam.read()
    #     image = imutils.rotate(image, 180)
    #     start_time = time.time()
    #     img = np.float32(image)  # [:, :, [2, 1, 0]].astype('float32')
    #     img_data = np.transpose(img, [2, 0, 1])
    #     img_data = np.expand_dims(img_data, axis=0)

    #     input_name = sess.get_inputs()[0].name

    #     output = sess.run(None, {input_name: img_data})
    #     predictions = demo_postprocess(output[0], input_shape, False)[0]

    #     boxes = predictions[:, :4]
    #     scores = predictions[:, 4:5] * predictions[:, 5:]

    #     boxes_xyxy = np.ones_like(boxes)
    #     boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    #     boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    #     boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    #     boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.

    #     dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    #     if dets is not None:
    #         final_boxes, final_scores, final_cls_inds = dets[:,
    #                                                          :4], dets[:, 4], dets[:, 5]
    #         image = vis(image, final_boxes, final_scores, final_cls_inds,
    #                     conf=0.3, class_names=CLASSES)
    #     cv2.putText(image, f"FPS: {round(1/(time.time()-start_time),2)}",
    #                 (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), thickness=1)
    #     sender.send_image(rpi_name,  image)
    #     print("end")
