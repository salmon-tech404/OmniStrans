import os
from pathlib import Path

def download_file(url, dest_path, zip_member, status_callback=None, model_name=""):
    import requests
    import io
    import zipfile
    
    if status_callback:
        status_callback(f"Đang tải model {model_name}... 0%")
        
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        
        zip_data = io.BytesIO()
        downloaded = 0
        last_percent = -1
        
        for chunk in r.iter_content(chunk_size=16384):
            if chunk:
                zip_data.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = int(downloaded * 100 / total_size)
                    if percent != last_percent:
                        last_percent = percent
                        if status_callback:
                            status_callback(f"Đang tải model {model_name}... {percent}%")
                            
        zip_data.seek(0)
        with zipfile.ZipFile(zip_data) as z:
            for name in z.namelist():
                if name.lower() == zip_member.lower() or name.lower().endswith(zip_member.lower()):
                    with open(dest_path, "wb") as f:
                        f.write(z.read(name))
                    return True
        raise Exception(f"Không tìm thấy {zip_member} trong file zip.")
    except Exception as e:
        raise Exception(f"Lỗi tải model {model_name}: {e}")

class OCRReader:
    def __init__(self, status_callback=None):
        # Xác định thư mục gốc của dự án
        base_dir = Path(__file__).parent.resolve()
        models_dir = base_dir / "models" / "ocr"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        craft_path = models_dir / "craft_mlt_25k.pth"
        latin_path = models_dir / "latin_g2.pth"
        
        # Tải CRAFT detector model nếu chưa có
        if not craft_path.exists():
            craft_url = "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip"
            download_file(craft_url, craft_path, "craft_mlt_25k.pth", status_callback, "phát hiện chữ (CRAFT)")
            
        # Tải Latin recognition model nếu chưa có
        if not latin_path.exists():
            latin_url = "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/latin_g2.zip"
            download_file(latin_url, latin_path, "latin_g2.pth", status_callback, "nhận diện chữ (Latin)")
            
        # Thư viện torch và easyocr rất nặng nên sẽ import lazy tại đây để tránh làm chậm startup app
        import torch
        import easyocr
        
        # Tự động phát hiện GPU/CUDA nếu có để tăng tốc nhận diện
        use_gpu = torch.cuda.is_available()
        if status_callback:
            status_callback("Đang khởi tạo EasyOCR Reader...")
        print(f"[EasyOCR] Khởi tạo Reader (GPU={use_gpu})")
        
        # Khởi tạo Reader hỗ trợ tiếng Việt ('vi') và tiếng Anh ('en')
        self.reader = easyocr.Reader(
            lang_list=['vi', 'en'], 
            gpu=use_gpu, 
            model_storage_directory=str(models_dir),
            download_enabled=False
        )

    def extract_text(self, image_path):
        """
        Trích xuất văn bản từ tệp ảnh bằng EasyOCR.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_path}")
            
        results = self.reader.readtext(image_path)
        # Kết hợp các đoạn text nhận diện được, phân tách bằng dấu xuống dòng
        texts = [res[1] for res in results if res[1]]
        return " ".join(texts).strip()
