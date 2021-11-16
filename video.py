#-------------------------------------#
#   调用摄像头或者视频进行检测
#   调用摄像头直接运行即可
#   调用视频可以将cv2.VideoCapture()指定路径
#   视频的保存并不难，可以百度一下看看
#-------------------------------------#
import time
import cv2

from hdtrack.detection import OnnxDetection,TorchDetection
from hdtrack.tracker import EvalTracker,EvalEmbed,DeepsortTracker,JdeTracker
from hdtrack.tracker import OnnxSdeEmbed,TorchSdeEmbed,OnnxJdeEmbed,TorchJdeEmbed

embed_model=TorchSdeEmbed(['motor'])
tracker=DeepsortTracker(embed_model)
#-------------------------------------#
#   调用摄像头
#   capture=cv2.VideoCapture("1.mp4")
#-------------------------------------#
capture=cv2.VideoCapture("D:/Dataset/indian_cctv_motor_helmet/c1.avi")
fps = 0.0
frame_id=0
while(True):
    t1 = time.time()
    # 读取某一帧
    ref,frame=capture.read()
    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # 进行检测
    frame = tracker.track_one_image(frame)
    # RGBtoBGR满足opencv显示格式

    fps  = ( fps + (1./(time.time()-t1)) ) / 2
    print("fps= %.2f"%(fps))
    #frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite('./results/'+str(frame_id)+'.jpg',frame)
    frame_id+=1
    #cv2.namedWindow("video",cv2.WINDOW_AUTOSIZE)
    #cv2.imshow("video",frame)

    c= cv2.waitKey(1) & 0xff # 每隔1ms捕获键盘
    if c==27: # 27是esc键
        capture.release()
        break
