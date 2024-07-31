import cv2
import os

def save_frames_from_video(video_path, output_folder, step):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    step_frames = int(fps * step)

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % step_frames == 0:
            print(f"Saving frame {frame_count}")
            output_path = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"Saved {saved_frame_count} frames to {output_folder}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str)
    parser.add_argument("--output-dir", default="./output", type=str)
    parser.add_argument("--step", default=10, type=int)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    save_frames_from_video(args.video_path, args.output_dir, args.step)
