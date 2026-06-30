# Hướng Dẫn Cài Đặt & Sử Dụng Tesseract OCR (Offline)

Tài liệu này hướng dẫn chi tiết cách cài đặt, cấu hình và sử dụng công cụ **Tesseract OCR** để quét văn bản trên màn hình ở chế độ ngoại tuyến (Offline) trong ứng dụng dịch thuật thời gian thực.

---

## 1. Giới Thiệu
Chế độ **Tesseract OCR (Offline)** cho phép ứng dụng nhận diện chữ trực tiếp từ hình ảnh chụp màn hình mà không cần gửi dữ liệu ảnh lên internet. Điều này giúp:
* Tiết kiệm băng thông internet.
* Tốc độ nhận diện nhanh, phản hồi tức thì.
* Bảo mật thông tin (không chia sẻ hình ảnh lên server đám mây).
* Hoạt động ổn định ngay cả khi không có kết nối internet hoặc khi tài khoản Gemini Web bị giới hạn/quá tải.

---

## 2. Các Bước Cài Đặt Tesseract OCR trên Windows

Để tính năng này hoạt động, bạn cần cài đặt phần mềm Tesseract OCR lên hệ điều hành Windows của mình.

### Bước 2.1: Tải bộ cài đặt Tesseract OCR
1. Truy cập vào trang tải xuống các bản dựng của Tesseract dành cho Windows (do UB Mannheim cung cấp):
   * Link tải: [Tesseract OCR W64 Builds (UB-Mannheim)](https://github.com/UB-Mannheim/tesseract/wiki)
2. Tải về phiên bản mới nhất phù hợp với Windows 64-bit (ví dụ: `tesseract-ocr-w64-setup-v5.x.x...exe`).

### Bước 2.2: Tiến hành cài đặt
1. Chạy file installer vừa tải về.
2. Chọn ngôn ngữ cài đặt (thường là English).
3. Tại bước **Choose Components**:
   * Bạn có thể mở rộng mục **Additional language data (download)** và tick chọn **Vietnamese** cùng **English** (hoặc các ngôn ngữ khác nếu muốn).
   * *Lưu ý: Ứng dụng đã được lập trình để tự động tải các tệp dữ liệu ngôn ngữ tiếng Anh và tiếng Việt (`eng.traineddata`, `vie.traineddata`) vào thư mục `tessdata` trong dự án. Do đó, việc tải thêm ở bước này là tùy chọn.*
4. **Đường dẫn cài đặt mặc định (Quan trọng):**
   * Hãy giữ nguyên đường dẫn mặc định: `C:\Program Files\Tesseract-OCR`
   * Nhấp **Next** và hoàn tất quá trình cài đặt.

---

## 3. Cấu Hình Đường Dẫn trong Mã Nguồn

Ứng dụng của chúng ta đã được cấu hình sẵn để tìm Tesseract tại đường dẫn mặc định trong tệp [ocr_processor.py](file:///c:/Users/HT/Desktop/realtime-translation/realtime-translation-new/ocr_processor.py):

```python
# Thiết lập đường dẫn tới file chạy của Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

> [!IMPORTANT]
> **Trường hợp bạn cài đặt Tesseract ở thư mục khác:**
> Bạn bắt buộc phải mở file [ocr_processor.py](file:///c:/Users/HT/Desktop/realtime-translation/realtime-translation-new/ocr_processor.py) và cập nhật lại đường dẫn chính xác tại dòng số 7. Ví dụ:
> ```python
> pytesseract.pytesseract.tesseract_cmd = r"D:\MyApps\Tesseract-OCR\tesseract.exe"
> ```

---

## 4. Dữ Liệu Ngôn Ngữ (Language Data)

Để Tesseract nhận diện chính xác tiếng Việt và tiếng Anh, hệ thống cần các tệp dữ liệu ngôn ngữ đặt trong thư mục `tessdata`.

* **Tự động tải xuống:** 
  Khi khởi chạy ứng dụng lần đầu tiên, chương trình sẽ tự động kiểm tra và tải xuống hai tệp dữ liệu huấn luyện:
  * `vie.traineddata` (nhận diện Tiếng Việt)
  * `eng.traineddata` (nhận diện Tiếng Anh)
  Các file này sẽ được lưu trữ cục bộ trong thư mục `tessdata` ngay tại gốc của thư mục ứng dụng.

* **Tải thủ công (Dự phòng):**
  Nếu việc tự động tải xuống gặp lỗi (do chặn tường lửa, mất kết nối...), bạn có thể tự tải thủ công:
  1. Tải [vie.traineddata](https://github.com/tesseract-ocr/tessdata_fast/raw/main/vie.traineddata) và [eng.traineddata](https://github.com/tesseract-ocr/tessdata_fast/raw/main/eng.traineddata).
  2. Tạo thư mục tên `tessdata` tại thư mục gốc của dự án này.
  3. Di chuyển cả hai tệp `.traineddata` vừa tải vào thư mục `tessdata`.

---

## 5. Hướng Dẫn Sử Dụng Trên Giao Diện Ứng Dụng

Sau khi hoàn tất cài đặt phần mềm Tesseract OCR trên máy tính:

1. **Chạy ứng dụng**: Mở chương trình chính bằng cách kích hoạt môi trường ảo và chạy `main.py` hoặc file thực thi (.exe) đã build.
2. **Chuyển tab**: Trên giao diện chính, chọn Tab **📷 Dịch hình ảnh (OCR)**.
3. **Chọn Chế độ**: Ở ô chọn chế độ OCR (Mode), chuyển từ `Gemini Web OCR (Online)` sang **`Tesseract OCR (Offline)`**.
4. **Thực hiện quét hình ảnh**:
   * Click vào nút **`📷 Quét vùng màn hình (OCR)`** hoặc nhấn phím tắt quét OCR đã cấu hình sẵn trong phần Cài đặt Hệ thống.
   * Con trỏ chuột sẽ chuyển thành dạng hồng tâm. Hãy nhấn giữ và kéo chuột để tạo một vùng bao quanh đoạn chữ bạn muốn dịch trên màn hình.
   * Thả chuột ra, ứng dụng sẽ ngay lập tức trích xuất chữ bằng Tesseract và dịch văn bản đó sang ngôn ngữ đích của bạn.
5. **Xem kết quả**: Kết quả sẽ hiển thị trong khung văn bản bên dưới dạng:
   ```text
   [ORIGINAL]
   <Văn bản gốc do Tesseract nhận diện>

   [TRANSLATED]
   <Bản dịch tương ứng>
   ```

---

## 6. Xử Lý Sự Cố Thường Gặp (Troubleshooting)

### Lỗi 1: `pytesseract.pytesseract.TesseractNotFoundError: tesseract is not installed or it's not in your PATH`
* **Nguyên nhân:** Ứng dụng không tìm thấy tệp `tesseract.exe` tại đường dẫn chỉ định.
* **Cách khắc phục:** 
  1. Kiểm tra xem bạn đã chạy bộ cài đặt Tesseract ở **Mục 2** chưa.
  2. Kiểm tra xem file `tesseract.exe` có thực sự nằm tại đường dẫn `C:\Program Files\Tesseract-OCR\tesseract.exe` hay không.
  3. Nếu cài ở ổ đĩa khác hoặc thư mục khác, hãy sửa lại đường dẫn trong file `ocr_processor.py` như hướng dẫn ở **Mục 3**.

### Lỗi 2: Kết quả quét trả về trống (`Không tìm thấy chữ trong vùng chọn`)
* **Nguyên nhân:**
  * Vùng quét không chứa chữ hoặc chữ quá mờ, độ tương phản quá thấp.
  * Thiếu file ngôn ngữ tiếng Việt/tiếng Anh trong thư mục `tessdata`.
* **Cách khắc phục:**
  * Thử quét vùng rộng hơn, đảm bảo chữ rõ nét và không bị che khuất.
  * Kiểm tra thư mục `tessdata` trong dự án xem đã có đủ hai tệp `eng.traineddata` và `vie.traineddata` chưa. Nếu chưa, tải thủ công như hướng dẫn ở **Mục 4**.

### Lỗi 3: Chữ nhận diện bị sai font hoặc hiển thị ký tự lạ
* **Nguyên nhân:** Tesseract đang sử dụng sai file ngôn ngữ hoặc font chữ trên màn hình quá đặc biệt.
* **Cách khắc phục:** 
  * Hãy đảm bảo đã cấu hình ngôn ngữ quét là `vie+eng` trong code (mặc định đã được cấu hình nhận diện cả hai ngôn ngữ đồng thời).
  * Chụp các chữ có thiết kế tiêu chuẩn, rõ ràng, tránh các font chữ nghệ thuật (handwriting/gothic) quá phức tạp.
