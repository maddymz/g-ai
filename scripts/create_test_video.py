"""Generate a simple test video for video_search testing."""
import cv2
import numpy as np

# Create a simple test video with colored frames
output_path = '../fixtures/videos/test_video.mp4'
fps = 30
duration = 2  # seconds
width, height = 320, 240

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

total_frames = fps * duration
for i in range(total_frames):
    # Create frame with gradient color change
    color_ratio = i / total_frames
    r = int(255 * (1 - color_ratio))
    g = int(255 * color_ratio)
    b = 128
    
    frame = np.ones((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = b  # Blue
    frame[:, :, 1] = g  # Green
    frame[:, :, 2] = r  # Red
    
    # Add frame number text
    cv2.putText(frame, f'Frame {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    out.write(frame)

out.release()
print(f"Created test video: {output_path} ({total_frames} frames, {duration}s)")
