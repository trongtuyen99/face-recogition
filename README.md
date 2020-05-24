# face-recogition using CNN
**nhận dạng khuôn mặt sử dụng cnn + opencv**
## dữ liệu:
  - sử dụng bộ dữ liệu của 21 người nổi tiếng, mỗi người 45 ảnh
  - augmentation sinh thêm ảnh: rotate, flip, add noise
## phương pháp:
  - sử dụng opencv detect khuôn mặt
  - sử dụng cnn nhận dạng khuôn mặt
    - độ chính xác trên tập train: 99%
    - độ chính xác trên tập test: 94.71%
## Chức năng:
  - nhận dạng khuôn mặt đầu vào là video quay bằng webcam / ảnh
## cài đặt:
1. môi trường:
  - python >= 3.5
  - pip install tensorflow opencv-python keras
2. chạy trương trình:
  1. giải nén file model
  2. cấu hình lại đường dẫn của các file _95.h5 và _95.json trong hàm main
  3. chạy hàm main để nhận dạng ảnh.
