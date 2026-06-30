import os
import urllib.request

def setup_tesseract(base_dir):
    """
    Configures Tesseract path environment.
    """
    tessdata_dir = os.path.join(base_dir, "tessdata")
    os.environ["TESSDATA_PREFIX"] = tessdata_dir
    return tessdata_dir

def download_traineddata_if_missing(tessdata_dir):
    """
    Downloads the vie.traineddata and eng.traineddata files if they do not exist locally.
    """
    os.makedirs(tessdata_dir, exist_ok=True)
    languages = {
        "vie.traineddata": "https://github.com/tesseract-ocr/tessdata_fast/raw/main/vie.traineddata",
        "eng.traineddata": "https://github.com/tesseract-ocr/tessdata_fast/raw/main/eng.traineddata"
    }
    
    for filename, url in languages.items():
        file_path = os.path.join(tessdata_dir, filename)
        if not os.path.exists(file_path):
            print(f"[OCR] Downloading {filename} for local Tesseract...")
            try:
                req = urllib.request.Request(
                    url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
                    out_file.write(response.read())
                print(f"[OCR] Downloaded {filename} successfully.")
            except Exception as e:
                print(f"[OCR Error] Error downloading {filename}: {e}")

def run_local_ocr(image_path):
    """
    Runs local Tesseract OCR on the image.
    """
    try:
        import pytesseract
        from PIL import Image
        
        # Set Tesseract executable path dynamically
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        
        img = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(img, lang="vie+eng")
        return extracted_text.strip() if extracted_text else ""
    except Exception as e:
        raise Exception(f"Lỗi Tesseract OCR: {str(e)}")
