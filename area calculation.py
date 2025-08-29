import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt

# 加载模型
model = YOLO('D:/ultralytics-main/PerovSegNet.pt')

# 加载图像
img_path = 'D:/ultralytics-main/000008.jpg'
image = cv2.imread(img_path)

# 进行预测
results = model.predict(image)

# 初始化计数器和总面积变量
count = 0
total_area = 0
pixel_to_nm2 = 1.271   # 每个像素的实际面积 (nm^2)
label2_info = []  # 用于存储标签2的检测信息

# 获取预测结果
for result in results:
    if hasattr(result.boxes, 'xyxy'):
        boxes = result.boxes.xyxy.cpu().numpy()  # 获取边界框坐标并从 GPU 转移到 CPU
        classes = result.boxes.cls.cpu().numpy()  # 获取类别标签
        for i, box in enumerate(boxes):
            class_id = int(classes[i])

            # 如果标签是标签2，进行处理
            if class_id == 0:  # 假设标签2的class_id为0，具体值根据模型调整
                count += 1
                x1, y1, x2, y2 = box[:4]
                area = (x2 - x1) * (y2 - y1)  # 计算面积
                actual_area = area * pixel_to_nm2  # 实际面积 (nm²)
                total_area += actual_area

                # 保存标签2的信息
                label2_info.append({
                    "id": count,  # 使用count作为编号
                    "area": actual_area,
                    "box": box
                })

                # 在图像上绘制标签2的边界框和编号
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(image, f"ID: {count}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 打印标签2的检测数量和总面积
print(f"检测到标签2的物体数量: {count}")
print(f"标签2检测区域的总面积: {total_area:.2f} nm^2")

# 创建一个新图像来显示面积列表
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 在左边列出标号和面积
ax[0].axis('off')  # 关闭左侧的坐标轴
ax[0].text(0.1, 1, 'ID and Area of PbI₂ objects', fontsize=12, fontweight='bold')
for idx, label in enumerate(label2_info, 1):
    ax[0].text(0.1, 1 - 0.022 * idx, f"ID: {label['id']}, Area: {label['area']:.2f} nm^2", fontsize=10)

# 在右侧显示预测的图像
ax[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[1].set_title(f"Detected {count} PbI₂ objects", fontsize=12)
ax[1].axis('off')  # 隐藏坐标轴

plt.tight_layout(pad=2.0)  # 调整布局以避免重叠
plt.show()
