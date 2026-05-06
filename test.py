from ultralytics import YOLO

# 加载训练好的模型权重
model = YOLO('D:/PyCharm/pyproject/simple-lama-inpainting-main/runs_yolo_watermark/yolov8n_watermark/weights/best.pt')

# 读入图片路径
img_path = 'D:/PyCharm/pyproject/simple-lama-inpainting-main/tests/data_self/image_2.jpg'

# 运行检测
results = model(img_path)  # conf置信度阈值，iou是NMS阈值

# 结果可视化
results[0].show()  # 会弹出窗口显示带框的图片（需图形界面）

# 保存结果带框的图片
# results.save(save_dir='runs_yolo_infer')

# 打印检测框信息
for result in results:
    print(result.boxes.data)  # 检测框坐标，置信度，类别等信息
