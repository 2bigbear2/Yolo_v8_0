from ultralytics import YOLO
import torch

# 检查CUDA是否可用
print(f"CUDA available: {torch.cuda.is_available()}")

# 如果可用，打印GPU名称和数量
if torch.cuda.is_available():
    print(f"GPU device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

# 加载模型
yolo = YOLO("./ultralytics_main/yolov8s.pt", task="detect")

# 进行预测（不指定device，看默认行为）
# 使用YOLO模型，对路径为 ./ultralytics/assets/bus.jpg 的图片进行目标检测。只保留置信度大于等于0.8的检测结果，并将带有检测框的结果图片保存下来。
result = yolo(source="./ultralytics_main/ultralytics/assets/bus.jpg", save=True, conf=0.8)