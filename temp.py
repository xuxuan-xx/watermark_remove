import cv2
import torch
import numpy as np
from PIL import Image
import time
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from simple_lama_inpainting import SimpleLama

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 加载YOLO检测模型（改成你的权重路径）
    yolo_model = YOLO("runs_yolo_watermark/yolov8n_watermark/weights/best.pt")

    # 加载SAM模型和初始化预测器
    sam_checkpoint = "D:/PyCharm/pyproject/simple-lama-inpainting-main/tests/sam_model/sam_vit_b_01ec64.pth"
    sam_model_type = "vit_b"
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)

    # 初始化LaMa修复模型
    lama_model = SimpleLama()

    # 视频输入输出路径
    video_input_path = "D:/PyCharm/pyproject/simple-lama-inpainting-main/test.mp4"
    video_output_path = "output_inpainted_video.mp4"

    cap = cv2.VideoCapture(video_input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. YOLO检测水印框
        results = yolo_model.predict(source=frame_rgb, conf=0.4, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()  # (N,4)格式

        if len(boxes) == 0:
            # 无水印，直接写出
            out.write(frame)
            print(f"帧 {frame_idx}: 无水印，跳过，耗时 {time.time() - start_time:.3f}s")
            frame_idx += 1
            continue

        # 2. 用SAM单独预测每个框掩码，合成最终mask
        predictor.set_image(frame_rgb)
        final_mask = np.zeros((h, w), dtype=np.uint8)

        for box in boxes:
            # SAM需要box格式为 [x1, y1, x2, y2]
            input_box = box.astype(int)
            masks, scores, logits = predictor.predict(
                box=input_box,
                multimask_output=False,
            )
            # masks shape: (1, H, W),取第一个mask
            mask = masks[0].astype(np.uint8) * 255
            final_mask = np.maximum(final_mask, mask)  # 合并掩码

        # 3. LaMa修复
        pil_img = Image.fromarray(frame_rgb)
        pil_mask = Image.fromarray(final_mask).convert("L")
        inpainted_pil = lama_model(pil_img, pil_mask)
        inpainted_bgr = cv2.cvtColor(np.array(inpainted_pil), cv2.COLOR_RGB2BGR)

        # 写出结果帧
        out.write(inpainted_bgr)

        print(f"帧 {frame_idx} 处理完成，总耗时: {time.time() - start_time:.3f}s")
        frame_idx += 1

    cap.release()
    out.release()
    print(f"🎉 视频处理完成，保存路径：{video_output_path}")

if __name__ == "__main__":
    main()
