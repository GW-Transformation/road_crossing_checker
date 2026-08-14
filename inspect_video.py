import cv2
import os

videos = ["/Users/m4ck/Desktop/1.mov", "/Users/m4ck/Desktop/2.mov"]
os.makedirs("/Users/m4ck/projects/road_crossing_checker/inspect_frames", exist_ok=True)

for vid_path in videos:
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    print(f"Video: {vid_path}")
    print(f"  Resolution: {width}x{height}, FPS: {fps:.2f}, Frames: {total_frames}, Duration: {duration:.2f}s")
    
    vid_name = os.path.basename(vid_path).split('.')[0]
    step = max(1, int(fps * 0.3))
    saved = 0
    for f_idx in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret:
            out_name = f"/Users/m4ck/projects/road_crossing_checker/inspect_frames/{vid_name}_frame_{f_idx:04d}_{f_idx/fps:.1f}s.jpg"
            cv2.imwrite(out_name, frame)
            saved += 1
    print(f"  Saved {saved} frames for inspection.")
    cap.release()
