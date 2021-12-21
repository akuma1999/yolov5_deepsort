from utils.my_module import predict_clf, remake_point
import numpy as np
import torch.backends.cudnn as cudnn
import torch
import cv2
from tensorflow.keras.models import load_model
from pathlib import Path
import sys
from models.experimental import attempt_load
from utils.datasets import LoadStreams
from utils.general import non_max_suppression,  scale_coords, set_logging, xyn2xy
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync
import os


from deep_sort_pytorch.deep_sort import DeepSort

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


FILE = Path(__file__).resolve()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

# load model
# model_gender = load_model("./h5/gender/weights-23-0.93.hdf5")
#


@torch.no_grad()
def run():

    # varaiable
    source = '0'
    imgsz = [416, 416]
    conf_thres = 0.4
    iou_thres = 0.45
    max_det = 1000
    device = ''
    classes = None
    agnostic_nms = False
    augment = False
    visualize = False
    # line_thickness = 3
    # hide_labels = False
    # hide_conf = False
    half = False
    w = './h5/mask/mark.best.last.pt'

    # Initialize
    set_logging()
    device = select_device(device)  # chọn phần cứng để sử dụng
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    model = attempt_load([w], map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(
        model, 'module') else model.names  # get class names

    # -----------------------------------------------------------------

    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=True)

    # Run inference
    if device.type != 'cpu':
        # run once
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, img, im0s, vid_cap in dataset:
        t1 = time_sync()

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

# -----------------------------------------------------------------
        # Inference

        pred = model(img, augment=augment, visualize=visualize)[0]

# --------------------------------------
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
          # batch_size >= 1
            p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(
            ), dataset.count

            p = Path(p)  # to Path

            s += '%gx%g ' % img.shape[2:]  # print string
            # normalization gain whwh
            # annotator = Annotator(
            #     im0, line_width=line_thickness, pil=not ascii)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    box = remake_point(xyxy)
                    x1 = box[0]
                    y1 = box[1]
                    x2 = box[2]
                    y2 = box[3]
                    border = x2 - x1
                    img_x = im0.copy()
                    imgBorder = cv2.copyMakeBorder(
                        img_x, border, border, border, border, cv2.BORDER_CONSTANT, value=[150, 150, 150])
                    img_scrop = imgBorder[
                        y1+border - int(border/3):y2 + border+int(border/3),
                        x1+border - int(border/3):x2 + border+int(border/3)
                    ]

                    im0 = predict_clf(img_scrop, box, model_gender, im0)
                    c = int(cls)  # integer class
                    # label = None if hide_labels else (
                    #     names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    # annotator.box_label(xyxy, label, color=colors(c, True))

            # Print time (inference-only)
            print('----------------------------')

            # print(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            # im0 = annotator.result()

            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond

    # Print results


def main():
    run()


if __name__ == "__main__":
    main()
