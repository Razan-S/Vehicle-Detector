import torch
import cv2
import os
from pathlib import Path

# Ensure the paths are correct
VIDEOS_DIR = Path('test-video')
video_path = VIDEOS_DIR / 'IMG_4799.mp4'
video_path_out = f"{VIDEOS_DIR}/{video_path.stem}_out.mp4"
output_images_dir = VIDEOS_DIR / 'output_images'
output_images_dir.mkdir(exist_ok=True)

cap = cv2.VideoCapture(str(video_path))

# Set custom FPS for output video
custom_fps = 15
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), custom_fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

model_path = r'yolov5-master\runs\train\exp4\weights\best.pt'

# Load a YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path))  # Load the custom YOLOv5 model

threshold = 0.5
frame_count = 0

# Calculate the total number of frames for the first 10 seconds
total_frames = custom_fps * 10

ret, frame = cap.read()
while ret and frame_count < total_frames:
    results = model(frame)  # Perform inference
    for result in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, model.names[int(class_id)].upper(), (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Save the frame as an image
    frame_image_path = output_images_dir / f"frame_{frame_count:04d}.jpg"
    cv2.imwrite(str(frame_image_path), frame)

    # Print the frame details
    print(f"Frame {frame_count}:")
    print(f"Image saved at: {frame_image_path}")
    print(f"Detection results: {results}")

    out.write(frame)
    ret, frame = cap.read()
    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()