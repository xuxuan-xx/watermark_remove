from simple_lama_inpainting import SimpleLama
from PIL import Image
import numpy as np

simple_lama = SimpleLama()

img_path = "D:/PyCharm/pyproject/simple-lama-inpainting-main/tests/data_self/image_3.jpg"
mask_path = "D:/PyCharm/pyproject/simple-lama-inpainting-main/tests/data_self/mask_3.jpg"

# 确保 image 是 RGB，mask 是灰度图
image = Image.open(img_path).convert("RGB")
mask = Image.open(mask_path).convert("L")

# LaMa 要求传入的是 3 通道图像 + 1 通道掩码（在内部拼接）
result = simple_lama(image, mask)
result.save("inpainted.jpg")
