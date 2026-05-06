from ultralytics import YOLO

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def train_yolov8():
    # 1. 配置
    yaml_path = "D:\PyCharm\pyproject\simple-lama-inpainting-main\yolo\watermark.yaml"  # 路径到配置文件
    model_name = "yolov8s.pt"          # 使用的预训练模型：n/s/m/l/x（轻量到重量）
    epochs = 100                        # 训练轮数
    img_size = 640                     # 输入图片尺寸
    project = "runs_yolo_watermark"    # 输出项目目录
    name = "yolov8n_watermark"         # 当前实验名称

    # 2. 创建 YOLO 模型
    model = YOLO(model_name)

    # 3. 开始训练
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=img_size,
        project=project,
        name=name,
        verbose=True,
        workers=0,
        batch=16,
        patience=20
    )

    # 4. 模型保存路径
    print(f"\n✅ 训练完成，最优模型保存在：{os.path.join(project, name, 'weights', 'best.pt')}")

if __name__ == "__main__":
    train_yolov8()



# 脚本指令： python train_yolo_watermark.py