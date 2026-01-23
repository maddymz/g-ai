"""Generate multiple test videos with different patterns for video search testing."""
import cv2
import numpy as np
import os

output_folder = '../fixtures/videos'
os.makedirs(output_folder, exist_ok=True)

fps = 30
duration = 2
width, height = 320, 240
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Video 1: Red to Blue gradient
out = cv2.VideoWriter(os.path.join(output_folder, 'red_blue_gradient.mp4'), fourcc, fps, (width, height))
for i in range(fps * duration):
    ratio = i / (fps * duration)
    r = int(255 * (1 - ratio))
    b = int(255 * ratio)
    frame = np.ones((height, width, 3), dtype=np.uint8)
    frame[:, :] = [b, 0, r]
    cv2.putText(frame, 'Red->Blue', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    out.write(frame)
out.release()
print("Created: red_blue_gradient.mp4")

# Video 2: Green to Yellow gradient (similar to red_blue in concept)
out = cv2.VideoWriter(os.path.join(output_folder, 'green_yellow.mp4'), fourcc, fps, (width, height))
for i in range(fps * duration):
    ratio = i / (fps * duration)
    g = 255
    r = int(255 * ratio)
    frame = np.ones((height, width, 3), dtype=np.uint8)
    frame[:, :] = [0, g, r]
    cv2.putText(frame, 'Green->Yellow', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    out.write(frame)
out.release()
print("Created: green_yellow.mp4")

# Video 3: Static blue
out = cv2.VideoWriter(os.path.join(output_folder, 'static_blue.mp4'), fourcc, fps, (width, height))
for i in range(fps * duration):
    frame = np.ones((height, width, 3), dtype=np.uint8)
    frame[:, :] = [255, 0, 0]  # Blue
    cv2.putText(frame, 'Static Blue', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    out.write(frame)
out.release()
print("Created: static_blue.mp4")

# Video 4: Checkerboard pattern (very different)
out = cv2.VideoWriter(os.path.join(output_folder, 'checkerboard.mp4'), fourcc, fps, (width, height))
for i in range(fps * duration):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    square_size = 20
    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            if (x // square_size + y // square_size) % 2 == 0:
                frame[y:y+square_size, x:x+square_size] = [255, 255, 255]
    cv2.putText(frame, 'Checker', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
    out.write(frame)
out.release()
print("Created: checkerboard.mp4")

print(f"\nCreated 4 test videos in {output_folder}/")
