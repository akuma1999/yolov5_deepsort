import cv2
import numpy as np


def predict_clf(img, box, model, img_org):
    img_test = img.copy()
    result_image = img_org.copy()
    img_test = cv2.resize(img_test, (128, 128))
    img_test = img_test.reshape(128, 128, 3)
    img_test = img_test.astype('float')*1./255
    img_test = np.expand_dims(img_test, axis=0)
    predict = model.predict(img_test)

    # custom font
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (0, 0, 255)
    thickness = 2
    #

    # map point
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    #

    result_image = cv2.rectangle(
        result_image, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)

    if predict[0][0] > 0.5:
        pre = round(predict[0][0]*100, 4)
        print('boy ', pre, " %")
        lable = f'boy: {str(pre)} %'
        result_image = cv2.putText(
            result_image, lable, (x1, y1), font, fontScale, color, thickness, cv2.LINE_AA)
    else:
        pre = round(100 - predict[0][0]*100, 4)
        print('girl ', pre, " %")
        lable = f'girl: {str(pre)} %'
        result_image = cv2.putText(
            result_image, lable, (x1, y1), font, fontScale, color, thickness, cv2.LINE_AA)
    return result_image


def remake_point(box):
    box_result = [int(i) for i in box]
    middle = ((box[2] - box[0]) - (box[3] - box[1]))/2
    if middle < 0:
        middle = abs(middle)
        box_result[0] = box_result[0] - int(middle)
        box_result[2] = box_result[2] + int(middle)
    else:
        box_result[1] = box_result[1] - int(middle)
        box_result[3] = box_result[3] + int(middle)
    return box_result
