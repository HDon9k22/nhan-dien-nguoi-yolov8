import cv2
import os
import face_recognition
from ultralytics import YOLO

# ✅ Đường dẫn file video
video_path = r"D:\Pycharm\PycharmProjects\DACN\video.nhan.dien2.mp4"

# ✅ Load mô hình YOLOv8 để nhận diện người
model = YOLO("yolov8n.pt")

# ✅ Tạo thư mục lưu ảnh người cắt từ video
output_folder = r"D:\Pycharm\PycharmProjects\DACN\cropped_people2"
os.makedirs(output_folder, exist_ok=True)

# ✅ Đọc video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Không thể mở video! Kiểm tra lại đường dẫn.")
    exit()

# ✅ Bộ trừ nền để phát hiện chuyển động
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

frame_count = 0  # Đếm số frame để xử lý từng khung hình
person_count = 0  # Đếm số người nhận diện

while True:
    ret, frame = cap.read() # đọc từng khung hình
    if not ret:
        break  # Kết thúc video

    frame_count += 1
    if frame_count % 3 != 0:
        continue  # Chỉ xử lý mỗi 3 frame để giảm tải

    # ✅ Phát hiện chuyển động
    fg_mask = bg_subtractor.apply(frame) # MOG2 tìm phần chuyển động
    _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # ✅ Tìm contour của vật thể chuyển động
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ✅ Nhận diện người bằng YOLO nếu có vật thể chuyển động
    results = model(frame)   # YOLO nhận diện người

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])  # Lớp object
            if cls == 0:  # Nếu là người
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])  # Lấy tọa độ bbox
                person_count += 1

                # ✅ Cắt ảnh người
                cropped_person = frame[y_min:y_max, x_min:x_max]

                # ✅ Kiểm tra nếu bounding box hợp lệ
                if cropped_person.size == 0:
                    continue

                # ✅ Resize ảnh về kích thước cố định (256x256)
                fixed_size = (256, 256)
                resized_person = cv2.resize(cropped_person, fixed_size)

                # ✅ Lưu ảnh
                output_path = os.path.join(output_folder, f"person_{person_count}.jpg")
                cv2.imwrite(output_path, resized_person)

                # ✅ Hiển thị kết quả
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ✅ Hiển thị video với vùng nhận diện
    cv2.imshow("Motion Detection & YOLO Person Detection", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
