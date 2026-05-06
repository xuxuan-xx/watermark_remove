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
clicked_point = []

# ===== 鼠标回调函数：点击图像中感兴趣的点 =====
def click_point(event, x, y, flags, param):
    global clicked_point, clone
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = [[x, y]]
        cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("点击水印上的任意点", clone)

cv2.namedWindow("点击水印上的任意点")
cv2.setMouseCallback("点击水印上的任意点", click_point)
cv2.imshow("点击水印上的任意点", clone)
cv2.waitKey(0)
cv2.destroyAllWindows()

if not clicked_point:
    print("❌ 未成功选点")
    exit()

# ===== 初始化 SAM 模型 =====
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)
predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

# ===== 准备 Point 提示输入 =====
input_point = np.array(clicked_point)
input_label = np.array([1])  # 表示这是前景点

# ===== SAM 预测（用点）=====
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False
)

# ===== 保存 mask =====
mask = (masks[0] * 255).astype(np.uint8)
cv2.imwrite(mask_output_path, mask)
print(f"✅ 点选掩码已保存到：{mask_output_path}")
