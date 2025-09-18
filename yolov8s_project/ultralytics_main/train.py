from ultralytics import YOLO
import os


def train_yolo():
    # 检查配置文件
    if not os.path.exists('yolo_test.yaml'):
        raise FileNotFoundError("数据集配置文件 yolo_test.yaml 不存在")

    # 检查AMP测试文件并处理
    amp_test_file = 'yolo11n.pt'
    use_amp = True  # 默认启用AMP

    if not os.path.exists(amp_test_file):
        print("AMP测试文件 yolo11n.pt 不存在")
        print("将禁用AMP功能继续训练")
        print("如需启用AMP，请手动下载:")
        print("https://github.com/ultralytics/assets/releases")
        use_amp = False  # 文件不存在则禁用AMP

    # 加载模型
    model = YOLO('yolov8s.pt')

    # 训练参数
    train_args = {
        'data': 'yolo_test.yaml',
        'epochs': 10,  # 增加训练轮数
        'batch': 5,  #一次处理多少图片
        'workers': 4,  # 根据CPU核心数调整（如4, 8）
        'device': '0',  # 指定GPU设备
        'val': True,  # 开启验证
        'amp': True,
    }

    # 开始训练
    results = model.train(**train_args)

    print("\n训练完成")

    return results


if __name__ == "__main__":
    train_yolo()


