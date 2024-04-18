from ultralytics import YOLO
import os
try:
    os.chdir('./loopy_detect_YOLOv8')
except:
    pass
#%% Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='./YOLO/loopy_detect_YOLOv8.yaml', epochs=200, imgsz=640,device=0)

#%%
model=YOLO('best.pt')

img_path = ""
results = model(img_path, show=True,save=False)
#%% 使用摄像头
import cv2
cap = cv2.VideoCapture(0)   # 0表示默认摄像头，如果有多个摄像头，可以尝试使用1, 2, 等

# 遍历视频帧
while cap.isOpened():
    # 从视频中读取一帧
    success, frame = cap.read()

    if success:
        # 在该帧上运行YOLOv8推理
        results = model(frame)

        # 在帧上可视化结果
        annotated_frame = results[0].plot()

        # 显示带注释的帧
        cv2.imshow("loopy_detect", annotated_frame)

        # 如果按下'q'则中断循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 如果视频结束则中断循环
        break

# 释放视频捕获对象并关闭显示窗口
cap.release()
cv2.destroyAllWindows()