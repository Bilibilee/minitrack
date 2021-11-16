import numpy as np
import cv2

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx+100) % 255, (17 * idx+100) % 255, (29 * idx+100) % 255)

    return color


def plot_results(image,class_names,prediction,is_print=False,color='rgb'):
    # 把bbox画在原图片上
    # is_print 表示是否打印出label和box值
    origin_image_shape = image.shape
    thickness = max(1, int(origin_image_shape[1] / 400.))
    fontscale=1
    if color !='rgb' and color!='bgr':
        raise ValueError
    if color=='rgb':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for obj in prediction:
        predicted_class = class_names[obj.label]
        score = obj.score

        left,top,right,bottom = obj.ltrb.astype(np.int32)
        if obj.track_id is  None:
            label_text ='{}'.format(predicted_class)  # '{} {:.2f}'.format(predicted_class, score)
            color = get_color(class_names.index(predicted_class))
        else:
            label_text = '{} ID-{}'.format(predicted_class, obj.track_id)
            color = get_color(obj.track_id)
        label_size=cv2.getTextSize(label_text,cv2.FONT_HERSHEY_SIMPLEX,fontScale=fontscale,thickness=thickness)
        if is_print==True:
            print(predicted_class, top, left, bottom, right)

        if top - label_size[1] >= 0:
            text_origin = (left, top - label_size[1])
        else:
            text_origin = (left, top + 1)

        image = cv2.putText(image,label_text,text_origin, cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontscale, color=color, thickness=thickness)
        image = cv2.rectangle(image, (left,top),(right,bottom), color=color, thickness=thickness)
    return image



