import cv2
import os
import face_recognition
from ultralytics import YOLO

# ✅ Đặt đường dẫn ảnh
main_subject_path = r"D:\Pycharm\PycharmProjects\DACN\DONG.JPG"  # Ảnh chủ thể
image_path = r"D:\Pycharm\PycharmProjects\DACN\image7.png"  # Ảnh chứa nhiều người
output_folder = r"D:\Pycharm\PycharmProjects\DACN\cropped_people"  # Thư mục lưu ảnh đã cắt
os.makedirs(output_folder, exist_ok=True)  # Tạo thư mục nếu chưa có

# ✅ Kiểm tra xem ảnh có tồn tại không
if not os.path.exists(image_path) or not os.path.exists(main_subject_path):
    print("❌ Ảnh không tồn tại, kiểm tra lại đường dẫn!")
    exit()

# ✅ Load YOLOv8
model = YOLO("yolov8n.pt")  # Mô hình nhận diện người

# ✅ Đọc ảnh gốc & ảnh chủ thể
image = cv2.imread(image_path)
subject_image = face_recognition.load_image_file(main_subject_path)

# ✅ Nhận diện khuôn mặt chủ thể
subject_encoding = face_recognition.face_encodings(subject_image)
if not subject_encoding:
    print("❌ Không tìm thấy khuôn mặt trong ảnh chủ thể! Vui lòng chọn ảnh khác.")
    exit()
subject_encoding = subject_encoding[0]  # Lấy encoding đầu tiên

# ✅ Nhận diện người trong ảnh
results = model(image)

# ✅ Biến lưu kết quả chính xác nhất
best_match = None
highest_similarity = 0

# ✅ Lặp qua tất cả các đối tượng được nhận diện
count = 0

for result in results:
    for box in result.boxes:
        if box.cls is not None and int(box.cls.item()) == 0:  # Chỉ nhận diện người
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

            # ✅ Nhận diện khuôn mặt trong ảnh cắt
            face_encodings = face_recognition.face_encodings(resized_person)

            # ✅ So sánh với chủ thể và tìm kết quả chính xác nhất
            if face_encodings:
                face_distance = face_recognition.face_distance([subject_encoding], face_encodings[0])[0]
                similarity_percentage = (1 - face_distance) * 100  # Tính phần trăm giống nhau

                # ✅ Cập nhật kết quả nếu độ chính xác cao hơn
                if similarity_percentage > highest_similarity:
                    highest_similarity = similarity_percentage
                    best_match = output_path

# ✅ Hiển thị kết quả cuối cùng
if best_match and highest_similarity >= 50:  # Chỉ hiển thị nếu giống trên 50%
    print(f"\n🎯 Chủ thể có thể là: {best_match} ({highest_similarity:.2f}% giống nhau)")
    img = cv2.imread(best_match)
    cv2.imshow("Best Match", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("❌ Không tìm thấy chủ thể với độ chính xác đủ cao!")
