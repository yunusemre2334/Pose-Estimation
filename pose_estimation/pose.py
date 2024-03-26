from ultralytics import YOLO
import cv2
import time


model_path = 'yolov8n-pose.pt'

cap = cv2.VideoCapture(0)

model = YOLO(model_path)

x1, y1, x2, y2 = 100, 100, 400, 400
x3, y3, x4, y4 = 300, 100, 400, 200
count_out = 0
count_in = 0
time_first_out = 0
time_first_in = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

    results = model(frame, verbose = False)[0]

    found_keypoints = False  

    for result in results:
        keypoints_np = result.keypoints.xy[0].numpy()
        for keypoint_indx, keypoint in enumerate(keypoints_np):
            x, y = map(int, keypoint[:2])
            cv2.putText(frame, str(keypoint_indx), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if x1 < x < x2 and y1 < y < y2:
                count_out += 1
                found_keypoints = True
                if x3 < x < x4 and y3 < y < y4:
                    count_in += 1


    if count_out > 4 and found_keypoints:
        if time_first_out == 0:
            time_first_out = time.time()
        if count_in > 2:
            if time_first_in == 0:
                time_first_in = time.time()
    else:
        time_first_out = 0 
        time_first_in = 0 

    if count_out <= 4:
        time_first_out = 0 

    if count_in <= 2:
        time_first_in = 0 

    if count_out > 5 and time_first_out != 0:
        time_end_out = time.time()
        elapsed_time = time_end_out - time_first_out
        print(f"Elapsed Time out: {elapsed_time:.2f} seconds")
        count_out = 0

    if count_in > 2 and time_first_in != 0:
        time_end_in = time.time()
        elapsed_time = time_end_in - time_first_in
        print(f"Elapsed Time in: {elapsed_time:.2f} seconds")
        count_in = 0

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
