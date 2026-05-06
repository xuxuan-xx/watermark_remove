import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

# ===== 配置路径 =====
checkpoint_path = "D:/PyCharm/pyproject/simple-lama-inpainting-main/tests/sam_model/sam_vit_h_4b8939.pth"  # 修改为你的路径
image_path = "D:/PyCharm/pyproject/simple-lama-inpainting-main/tests/data_self/image_3.jpg"          # 修改为你的图片路径
mask_output_path = "D:/PyCharm/pyproject/simple-lama-inpainting-main/tests/data_self/mask_3.jpg"
model_type = "vit_h"

# ===== 加载图像 =====
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
clone = image_bgr.copy()
bbox = []
drawing = False

# ===== 鼠标回调函数：框选水印区域 =====
def draw_rectangle(event, x, y, flags, param):
    global bbox, drawing, clone
    if event == cv2.EVENT_LBUTTONDOWN:
        bbox = [[x, y]]
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp = clone.copy()
        cv2.rectangle(temp, tuple(bbox[0]), (x, y), (0, 255, 0), 2)
        cv2.imshow("框选水印区域", temp)
    elif event == cv2.EVENT_LBUTTONUP:
        bbox.append([x, y])
        drawing = False
        cv2.rectangle(clone, tuple(bbox[0]), tuple(bbox[1]), (0, 255, 0), 2)
        cv2.imshow("框选水印区域", clone)

cv2.namedWindow("框选水印区域")
cv2.setMouseCallback("框选水印区域", draw_rectangle)
cv2.imshow("框选水印区域", image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(bbox) != 2:
    print("❌ 未成功框选区域")
    exit()

# ===== 准备框坐标 =====
x0, y0 = bbox[0]
x1, y1 = bbox[1]
box_np = np.array([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)])

# ===== 初始化 SAM 模型 =====
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)
predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

# ===== SAM 预测（用框）=====
masks, _, _ = predictor.predict(
    box=box_np,
    multimask_output=False
)

# ===== 保存 mask =====
mask = (masks[0] * 255).astype(np.uint8)
cv2.imwrite(mask_output_path, mask)
print(f"✅ Mask 已保存到：{mask_output_path}")
