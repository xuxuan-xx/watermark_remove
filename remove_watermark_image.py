import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from simple_lama_inpainting import SimpleLama

"""
cd simple-lama-inpainting-main
python remove_watermark_image.py
"""
# 路径设置
image_path = "D:/PyCharm/pyproject/simple-lama-inpainting-main/tests/data_self/test_1.jpg"
yolo_weight_path = "runs_yolo_watermark/yolov8n_watermark/weights/best.pt"
sam_checkpoint_path = "tests/sam_model/sam_vit_l_0b3195.pth"  # 使用较小的 vit_b 模型
output_path = "output_inpainted_image.jpg"


# 设备选择
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 加载模型
yolo_model = YOLO(yolo_weight_path)
sam = sam_model_registry["vit_l"](checkpoint=sam_checkpoint_path)
sam.to(device)
predictor = SamPredictor(sam)
lama_model = SimpleLama()

# 加载图像
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
h, w = image_rgb.shape[:2]

# 1. YOLOv8 检测
results = yolo_model.predict(source=image_rgb, conf=0.4, verbose=False)
boxes = results[0].boxes.xyxy.cpu().numpy()

if len(boxes) == 0:
    print("未检测到水印。")
    exit()

# 2. SAM 掩码生成
predictor.set_image(image_rgb)
final_mask = np.zeros((h, w), dtype=np.uint8)

for box in boxes:
    input_box = box.astype(int)
    masks, _, _ = predictor.predict(box=input_box, multimask_output=False)
    mask = masks[0].astype(np.uint8) * 255
    final_mask = np.maximum(final_mask, mask)

# 3. LaMa 修复
pil_image = Image.fromarray(image_rgb)
pil_mask = Image.fromarray(final_mask).convert("L")
inpainted = lama_model(pil_image, pil_mask)
inpainted_bgr = cv2.cvtColor(np.array(inpainted), cv2.COLOR_RGB2BGR)

# 保存结果
cv2.imwrite(output_path, inpainted_bgr)
print(f"✅ 去水印完成，结果已保存至：{output_path}")
