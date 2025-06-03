import cv2
import torch
import os
from ultralytics import YOLO

# ✅ Đặt đường dẫn ảnh
image_path = r"D:\Pycharm\PycharmProjects\DACN\image5.png"  # Ảnh gốc
output_folder = r"D:\Pycharm\PycharmProjects\DACN\cropped_people"  # Thư mục lưu ảnh đã cắt
os.makedirs(output_folder, exist_ok=True)  # Tạo thư mục nếu chưa có

# ✅ Kiểm tra xem ảnh có tồn tại không
if not os.path.exists(image_path):
    print(f"❌ Ảnh không tồn tại tại: {image_path}")
    exit()

# ✅ Load mô hình YOLOv8 (đã huấn luyện nhận diện người)
model = YOLO("yolov8n.pt")  # Chọn mô hình nhỏ nhất để chạy nhanh

# ✅ Đọc ảnh
image = cv2.imread(image_path)

# ✅ Nhận diện người trong ảnh
results = model(image)

# ✅ Lặp qua tất cả các đối tượng được nhận diện
count = 0
for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])  # Lấy chỉ số lớp
        if cls == 0:  # Lớp "0" là con người
            count += 1
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])  # Lấy tọa độ

            # ✅ Cắt ảnh người
            cropped_person = image[y_min:y_max, x_min:x_max]

            # ✅ Kiểm tra nếu bounding box hợp lệ
            if cropped_person.size == 0:
                print(f"❌ Bounding box trống cho người {count}. Không thể cắt ảnh!")
                continue

            # ✅ Resize về kích thước cố định (256x256)
            fixed_size = (256, 256)
            resized_person = cv2.resize(cropped_person, fixed_size)

            # ✅ Lưu ảnh từng người
            output_path = os.path.join(output_folder, f"cropped_person_{count}.jpg")
            cv2.imwrite(output_path, resized_person)
            print(f"✅ Ảnh người {count} đã được lưu tại: {output_path}")

            # ✅ Hiển thị ảnh đã cắt
            cv2.imshow(f"Person {count}", resized_person)

cv2.waitKey(0)
cv2.destroyAllWindows()

if count == 0:
    print("❌ Không phát hiện con người trong ảnh!")
else:
    print(f"🎯 Tổng số người được nhận diện và lưu: {count}")
