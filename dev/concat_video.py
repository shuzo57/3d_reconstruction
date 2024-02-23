import cv2
from ultralytics import YOLO

model_path = "/home/ohwada/3d_reconstruction/models/club-v2.pt"

def main(video_path1, video_path2, delay_frame, output_path):
    model = YOLO(model_path)
    
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)
    
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Video not opened")
    
    if delay_frame > 0:
        for i in range(delay_frame):
            ret, frame = cap1.read()
            if not ret:
                print("Error: Frame not read")
    elif delay_frame < 0:
        for i in range(-delay_frame):
            ret, frame = cap2.read()
            if not ret:
                print("Error: Frame not read")
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap1.get(cv2.CAP_PROP_FPS)
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width*2, height))
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if ret1 and ret2:
            results1 = model(frame1, verbose=False)
            results2 = model(frame2, verbose=False)
            
            frame1 = results1[0].plot()
            frame2 = results2[0].plot()
            
            frame = cv2.hconcat([frame1, frame2])
            out.write(frame)
        else:
            break
    
    cap1.release()
    cap2.release()
    out.release()

if __name__ == "__main__":
    video_path1 = "/home/ohwada/golf/2023_11_01/user1/d1/1/cam1_1_350_650.mp4"
    video_path2 = "/home/ohwada/golf/2023_11_01/user1/d1/1/cam2_1_300_600.mp4"
    delay_frame = -15
    output_path = "output.mp4"
    
    main(video_path1, video_path2, delay_frame, output_path)