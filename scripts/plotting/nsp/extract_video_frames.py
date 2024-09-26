import cv2
import os

def save_frames_from_video(video_file, output_dir, interval):
    # Open the video file
    cap = cv2.VideoCapture(video_file)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file}")
        return
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    frame_count = 0
    saved_frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Save the frame at the specified interval
        if frame_count % interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_frame_count:04d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
        
        frame_count += 1
    
    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved_frame_count} frames to {output_dir}")

# Example usage
prefix = "scripts/plotting/nsp/results/"
video_file = prefix + "pybullet_cover_typed_options_task2.mp4"
output_dir = prefix + "pybullet_cover_typed_options_task2_frames"
interval = 10  # Save every 10th frame

save_frames_from_video(video_file, output_dir, interval)