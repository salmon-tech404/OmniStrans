import sys
import queue
import threading
import os
import wave
import tempfile
import json
import asyncio
import subprocess
import time
import re
from pathlib import Path

# Add local paths
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
    RESOURCE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).parent
    RESOURCE_DIR = BASE_DIR

sys.path.insert(0, str(RESOURCE_DIR))

get_gemini_cookies = None
GeminiClient = None

if sys.stdout is not None:
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr is not None:
    sys.stderr.reconfigure(encoding='utf-8')

# ===============================
# TESSERACT OCR MODULE INTEGRATION
# ===============================
try:
    import ocr_processor
    # Setup Tesseract data path
    tessdata_dir = ocr_processor.setup_tesseract(str(BASE_DIR))
    # Run downloader in background
    threading.Thread(
        target=ocr_processor.download_traineddata_if_missing, 
        args=(tessdata_dir,), 
        daemon=True
    ).start()
except ImportError as e:
    print(f"[Import Warning] Could not import ocr_processor: {e}")
    ocr_processor = None
    tessdata_dir = None

# ===============================
# EASYOCR MODULE INTEGRATION
# ===============================
try:
    from EasyOCR import OCRReader
    easyocr_reader = None
except ImportError as e:
    print(f"[Import Warning] Could not import EasyOCR: {e}")
    OCRReader = None
    easyocr_reader = None

import numpy as np
import pyaudiowpatch as pyaudio


from PyQt6.QtCore import Qt, QTimer, QFileSystemWatcher, pyqtSignal, QRect, QPoint, QSize, QThread
from PyQt6.QtGui import QIcon, QAction, QColor, QPen, QPainter, QPixmap, QGuiApplication, QKeySequence, QShortcut, QFont, QCursor
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QTextEdit, QTextBrowser, QToolTip, QFrame, QComboBox, QSizeGrip, QFileDialog, QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QMessageBox, QCheckBox, QSystemTrayIcon, QMenu, QTabWidget, QScrollArea, QProgressBar

# ===============================
# DRACULA THEME COLORS
# ===============================
DRACULA = {
    "background": "#1A1C23", # Premium Slate Graphite Dark Background
    "background_alpha": "rgba(26, 28, 35, 230)", # Semi-transparent background
    "current_line": "#2B2D3A", # Slate active selection / input background
    "foreground": "#E5E9F0", # High contrast soft white (Nord/Slate)
    "comment": "#6C7A9C", # Muted gray-blue for secondary labels
    "cyan": "#5E81AC", # Steel-blue accent
    "green": "#A3BE8C", # Nord/Slate pastel green
    "green_rgb": "163, 190, 140",
    "orange": "#D08770", # Soft orange/salmon
    "pink": "#B48EAD", # Soft purple/lavender
    "purple": "#88C0D0", # Frost cyan for highlights
    "red": "#BF616A", # Soft red for error states
    "red_rgb": "191, 97, 106",
    "yellow": "#EBCB8B", # Warm yellow for source text
    
    # Button colors (Slate Green & Slate Red)
    "btn_green": "#4A9F5E",
    "btn_green_hover": "#5CAE6F",
    "btn_green_rgb": "74, 159, 94",
    "btn_red": "#BF616A",
    "btn_red_hover": "#C9717A",
    "btn_red_rgb": "191, 97, 106"
}

# ===============================
# CONFIG
# ===============================

# CHẾ ĐỘ THU ÂM (RECORD_MODE):
# - "loopback": Thu trực tiếp âm thanh phát ra LOA MẶC ĐỊNH của máy tính.
# - "mic": Thu âm từ MICROPHONE mặc định của hệ thống.
RECORD_MODE = "loopback"
SAMPLE_RATE = 16000
BLOCK_SECONDS = 4
SILENCE_THRESHOLD = 0.005
USE_GEMINI = True
DELETE_GEMINI_HISTORY = False

# CHẾ ĐỘ MẶC ĐỊNH KHI MỞ APP (Mặc định Online):
DEFAULT_WHISPER_MODE = "online"
DEFAULT_LOCAL_MODEL = "small"

WHISPER_MODE = DEFAULT_WHISPER_MODE
LOCAL_MODEL_NAME = DEFAULT_LOCAL_MODEL

# URL Server STT Online (chỉ dùng khi WHISPER_MODE = "online")
API_URL = "https://overremiss-kamron-uninquisitively.ngrok-free.dev/stt"

# CẤU HÌNH TẢI XUỐNG (yt-dlp & ffmpeg)
FFMPEG_PATH = ""
YT_DLP_PATH = ""
DL_COOKIE_VAL = ""

# Danh sách các ngôn ngữ hỗ trợ dịch và nhận diện
LANGUAGES = {
    "Tiếng Anh (English)": {"google": "en", "whisper": "en", "name_vi": "tiếng Anh", "name_en": "English"},
    "Tiếng Việt (Vietnamese)": {"google": "vi", "whisper": "vi", "name_vi": "tiếng Việt", "name_en": "Vietnamese"},
    "Tự động (Auto)": {"google": "auto", "whisper": None, "name_vi": "tự động phát hiện", "name_en": "Auto"},
    "Tiếng Nhật (Japanese)": {"google": "ja", "whisper": "ja", "name_vi": "tiếng Nhật", "name_en": "Japanese"},
    "Tiếng Trung (Chinese)": {"google": "zh-CN", "whisper": "zh", "name_vi": "tiếng Trung", "name_en": "Chinese"},
    "Tiếng Hàn (Korean)": {"google": "ko", "whisper": "ko", "name_vi": "tiếng Hàn", "name_en": "Korean"},
    "Tiếng Pháp (French)": {"google": "fr", "whisper": "fr", "name_vi": "tiếng Pháp", "name_en": "French"},
    "Tiếng Tây Ban Nha (Spanish)": {"google": "es", "whisper": "es", "name_vi": "tiếng Tây Ban Nha", "name_en": "Spanish"},
    "Tiếng Đức (German)": {"google": "de", "whisper": "de", "name_vi": "tiếng Đức", "name_en": "German"},
    "None (Không dịch)": {"google": "none", "whisper": None, "name_vi": "không dịch", "name_en": "None"}
}

SOURCE_LANG = "Tiếng Anh (English)"
TARGET_LANG = "Tiếng Việt (Vietnamese)"
DEEPSEEK_API_KEY = ""
GEMINI_API_KEY = ""
CURRENT_CONTEXT_PROMPT = ""

# Giao diện & Chủ đề cấu hình mặc định
THEME_MODE = "dark"
FONT_FAMILY_SRC = "'Segoe UI', sans-serif"
FONT_FAMILY_TGT = "'Segoe UI', sans-serif"
FONT_SIZE_SRC = 15
FONT_SIZE_TGT = 17

def get_clean_lang_name(full_name: str) -> str:
    if "Tự động" in full_name:
        return "Tự động"
    name = full_name.split("(")[0].strip()
    if name.startswith("Tiếng "):
        name = name[6:]
    return name

def load_settings() -> dict:
    global API_URL, SOURCE_LANG, TARGET_LANG, RECORD_MODE, SAMPLE_RATE, BLOCK_SECONDS, SILENCE_THRESHOLD, USE_GEMINI, FFMPEG_PATH, YT_DLP_PATH, DL_COOKIE_VAL, DELETE_GEMINI_HISTORY, DEEPSEEK_API_KEY, GEMINI_API_KEY, THEME_MODE, FONT_FAMILY_SRC, FONT_FAMILY_TGT, FONT_SIZE_SRC, FONT_SIZE_TGT, WHISPER_MODE, LOCAL_MODEL_NAME, CURRENT_CONTEXT_PROMPT
    config_path = BASE_DIR / "settings.json"
    settings = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
                API_URL = settings.get("api_url", API_URL)
                SOURCE_LANG = settings.get("source_lang", SOURCE_LANG)
                TARGET_LANG = settings.get("target_lang", TARGET_LANG)
                RECORD_MODE = settings.get("record_mode", RECORD_MODE)
                SAMPLE_RATE = int(settings.get("sample_rate", SAMPLE_RATE))
                BLOCK_SECONDS = int(settings.get("block_seconds", BLOCK_SECONDS))
                SILENCE_THRESHOLD = float(settings.get("silence_threshold", SILENCE_THRESHOLD))
                USE_GEMINI = settings.get("use_gemini", USE_GEMINI)
                FFMPEG_PATH = settings.get("ffmpeg_path", FFMPEG_PATH)
                YT_DLP_PATH = settings.get("yt_dlp_path", YT_DLP_PATH)
                DL_COOKIE_VAL = settings.get("dl_cookie_val", "")
                DELETE_GEMINI_HISTORY = settings.get("delete_gemini_history", DELETE_GEMINI_HISTORY)
                DEEPSEEK_API_KEY = settings.get("deepseek_api_key", "")
                GEMINI_API_KEY = settings.get("gemini_api_key", "")
                THEME_MODE = settings.get("theme_mode", THEME_MODE)
                FONT_FAMILY_SRC = settings.get("font_family_src", FONT_FAMILY_SRC)
                FONT_FAMILY_TGT = settings.get("font_family_tgt", FONT_FAMILY_TGT)
                FONT_SIZE_SRC = int(settings.get("font_size_src", FONT_SIZE_SRC))
                FONT_SIZE_TGT = int(settings.get("font_size_tgt", FONT_SIZE_TGT))
                WHISPER_MODE = settings.get("whisper_mode", WHISPER_MODE)
                LOCAL_MODEL_NAME = settings.get("local_model_name", LOCAL_MODEL_NAME)
                CURRENT_CONTEXT_PROMPT = settings.get("meeting_context_prompt", "")
        except Exception as e:
            print(f"Lỗi đọc cấu hình settings: {e}")
    return settings

def save_settings(data: dict):
    config_path = BASE_DIR / "settings.json"
    try:
        existing_data = {}
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except Exception as e:
                print(f"Lỗi đọc settings cũ để ghi đè: {e}")
        existing_data.update(data)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Lỗi ghi cấu hình settings: {e}")

# Khởi tạo cài đặt lúc chạy ứng dụng
initial_settings = load_settings()
API_URL = initial_settings.get("api_url", API_URL)
DELETE_GEMINI_HISTORY = initial_settings.get("delete_gemini_history", DELETE_GEMINI_HISTORY)
DEEPSEEK_API_KEY = initial_settings.get("deepseek_api_key", DEEPSEEK_API_KEY)
GEMINI_API_KEY = initial_settings.get("gemini_api_key", GEMINI_API_KEY)
THEME_MODE = initial_settings.get("theme_mode", THEME_MODE)
FONT_FAMILY_SRC = initial_settings.get("font_family_src", FONT_FAMILY_SRC)
FONT_FAMILY_TGT = initial_settings.get("font_family_tgt", FONT_FAMILY_TGT)
FONT_SIZE_SRC = int(initial_settings.get("font_size_src", FONT_SIZE_SRC))
FONT_SIZE_TGT = int(initial_settings.get("font_size_tgt", FONT_SIZE_TGT))
WHISPER_MODE = initial_settings.get("whisper_mode", WHISPER_MODE)
LOCAL_MODEL_NAME = initial_settings.get("local_model_name", LOCAL_MODEL_NAME)
CURRENT_CONTEXT_PROMPT = initial_settings.get("meeting_context_prompt", "")

# Trạng thái ghi âm (Mặc định tạm dừng cho đến khi người dùng bấm Bắt đầu)
IS_LISTENING = False

# Tên model Whisper cục bộ (chỉ dùng khi WHISPER_MODE = "local")
# Các model: tiny, base, small, medium, large-v3

# ===============================
# QUEUES
# ===============================

audio_queue = queue.Queue()
subtitle_queue = queue.Queue()

# ===============================
# TRANSLATOR
# ===============================

translator = None

def update_translation_config():
    global translator
    src_code = LANGUAGES.get(SOURCE_LANG, {}).get("google", "auto")
    tgt_code = LANGUAGES.get(TARGET_LANG, {}).get("google", "vi")
    try:
        if tgt_code == "none":
            translator = None
            print("[Translator] Dịch: Tắt")
        else:
            from deep_translator import GoogleTranslator
            translator = GoogleTranslator(source=src_code, target=tgt_code)
            print(f"[Translator] Dịch: {src_code} -> {tgt_code}")
    except Exception as e:
        print(f"[Translator Error] Lỗi cấu hình GoogleTranslator: {e}")

update_translation_config()

# ===============================
# GEMINI WEB CLIENT & TRANSLATION SETUP
# ===============================

gemini_loop = asyncio.new_event_loop()

def run_gemini_loop():
    asyncio.set_event_loop(gemini_loop)
    gemini_loop.run_forever()

threading.Thread(target=run_gemini_loop, daemon=True).start()

gemini_client = None
gemini_chat = None
active_gemini_profile = None  # Tên profile đang sử dụng
gemini_lock = None

async def async_init_gemini_client(profile_name: str) -> tuple[bool, str]:
    """Khởi tạo Gemini Client bất đồng bộ cho một profile."""
    global gemini_client, gemini_chat, active_gemini_profile, gemini_lock, get_gemini_cookies, GeminiClient
    
    if GeminiClient is None or get_gemini_cookies is None:
        try:
            from cookie_reader import get_gemini_cookies
            from gemini_webapi import GeminiClient
        except ImportError as e:
            print(f"[Import Warning] Could not import gemini_webapi or cookie_reader: {e}")
            get_gemini_cookies = None
            GeminiClient = None
            
    if not GeminiClient or not get_gemini_cookies:
        return False, "Không tìm thấy thư viện gemini_webapi hoặc cookie_reader"
        
    profile_path = BASE_DIR / "profiles" / profile_name
    if not profile_path.exists():
        return False, f"Thư mục profile không tồn tại: {profile_path}"
        
    try:
        psid, psidts = get_gemini_cookies(profile_path)
        if not psid:
            return False, "Không tìm thấy cookie __Secure-1PSID. Vui lòng đăng nhập lại."
            
        # Đóng client cũ nếu có
        if gemini_client:
            try:
                await gemini_client.close()
            except:
                pass
                
        gemini_client = GeminiClient(secure_1psid=psid, secure_1psidts=psidts or "", auto_close=False)
        await gemini_client.init(timeout=30, auto_refresh=True)
        gemini_chat = gemini_client.start_chat()
        gemini_lock = asyncio.Lock()
        active_gemini_profile = profile_name
        return True, f"Kết nối thành công profile: {profile_name}"
    except Exception as e:
        gemini_client = None
        gemini_chat = None
        active_gemini_profile = None
        gemini_lock = None
        return False, f"Lỗi khởi tạo: {str(e)}"

async def async_translate_with_gemini(text: str) -> str | None:
    """Dịch văn bản bất đồng bộ qua Gemini Chat."""
    global gemini_client, gemini_chat, gemini_lock, DELETE_GEMINI_HISTORY
    if not gemini_chat or not gemini_client:
        return None
    if gemini_lock is None:
        gemini_lock = asyncio.Lock()
    src_name = LANGUAGES.get(SOURCE_LANG, {}).get("name_vi", "tự động phát hiện")
    tgt_name = LANGUAGES.get(TARGET_LANG, {}).get("name_vi", "tiếng Việt")
    
    system_msg = (
        f"Hãy dịch câu sau từ {src_name} sang {tgt_name} tự nhiên và trôi chảy. "
        "Chỉ trả về bản dịch duy nhất, không thêm bất kỳ lời giải thích hay ký tự thừa nào khác."
    )
    if CURRENT_CONTEXT_PROMPT:
        system_msg += f"\n\nTuân thủ ngữ cảnh cuộc họp và quy tắc thuật ngữ sau:\n{CURRENT_CONTEXT_PROMPT}"
        
    prompt = (
        f"{system_msg}\n\n"
        f"Câu cần dịch:\n{text}"
    )
    async with gemini_lock:
        translate_chat = None
        try:
            translate_chat = gemini_client.start_chat() if DELETE_GEMINI_HISTORY else gemini_chat
            response = await translate_chat.send_message(prompt, temporary=DELETE_GEMINI_HISTORY)
            result_text = response.text.strip()
            if DELETE_GEMINI_HISTORY and translate_chat.cid:
                try:
                    await gemini_client.delete_chat(translate_chat.cid)
                except Exception as delete_err:
                    print(f"[Gemini Translation] Lỗi xóa history chat: {delete_err}")
            return result_text
        except Exception as e:
            print(f"[Gemini Send Message Error] {e}")
            if DELETE_GEMINI_HISTORY and translate_chat and translate_chat.cid:
                try:
                    await gemini_client.delete_chat(translate_chat.cid)
                except Exception as delete_err:
                    print(f"[Gemini Translation] Lỗi xóa history chat sau lỗi: {delete_err}")
            return None

def translate_with_gemini(text: str) -> str | None:
    """Dịch văn bản đồng bộ bằng cách chạy coroutine trên loop nền."""
    if not gemini_chat:
        return None
    future = asyncio.run_coroutine_threadsafe(async_translate_with_gemini(text), gemini_loop)
    try:
        return future.result(timeout=15)
    except Exception as e:
        print(f"[Gemini Web Translation Error] {e}")
        return None

async def async_transcribe_with_gemini_web(file_path: str) -> str | None:
    """Chép lại nội dung âm thanh của file bất đồng bộ qua Gemini Chat."""
    global gemini_client, gemini_chat, gemini_lock, DELETE_GEMINI_HISTORY
    if not gemini_chat or not gemini_client:
        return None
    if gemini_lock is None:
        gemini_lock = asyncio.Lock()
    
    src_name = LANGUAGES.get(SOURCE_LANG, {}).get("name_vi", "tự động phát hiện")
    prompt = (
        f"Hãy nghe file âm thanh đính kèm này và chép lại (transcribe) chính xác toàn bộ nội dung lời nói bằng {src_name}. "
        "Chỉ trả về nội dung văn bản chép lại duy nhất, tuyệt đối không thêm lời giải thích, tiêu đề, bình luận hoặc ký tự thừa nào khác."
    )
    async with gemini_lock:
        trans_chat = None
        try:
            trans_chat = gemini_client.start_chat() if DELETE_GEMINI_HISTORY else gemini_chat
            response = await trans_chat.send_message(prompt, files=[file_path], temporary=DELETE_GEMINI_HISTORY)
            result_text = response.text.strip()
            if DELETE_GEMINI_HISTORY and trans_chat.cid:
                try:
                    await gemini_client.delete_chat(trans_chat.cid)
                except Exception as delete_err:
                    print(f"[Gemini Transcribe] Lỗi xóa history chat: {delete_err}")
            return result_text
        except Exception as e:
            print(f"[Gemini Transcribe Error] {e}")
            if DELETE_GEMINI_HISTORY and trans_chat and trans_chat.cid:
                try:
                    await gemini_client.delete_chat(trans_chat.cid)
                except Exception as delete_err:
                    print(f"[Gemini Transcribe] Lỗi xóa history chat sau lỗi: {delete_err}")
            return None

def transcribe_with_gemini_web(file_path: str) -> str | None:
    """Chép lại nội dung âm thanh đồng bộ bằng cách chạy coroutine trên loop nền."""
    if not gemini_chat:
        return None
    future = asyncio.run_coroutine_threadsafe(async_transcribe_with_gemini_web(file_path), gemini_loop)
    try:
        return future.result(timeout=30)
    except Exception as e:
        print(f"[Gemini Web Transcribe Error] {e}")
        return None

async def async_ocr_with_gemini_web(image_path: str, translate: bool = True) -> str | None:
    """Nhận diện chữ (OCR) và dịch hình ảnh qua Gemini Web."""
    global gemini_client, gemini_chat, gemini_lock, DELETE_GEMINI_HISTORY
    if not gemini_client:
        return "ERROR: Chưa kết nối Gemini Web. Vui lòng kết nối Profile ở nút hình người 👤."
    if gemini_lock is None:
        gemini_lock = asyncio.Lock()
    
    src_name = LANGUAGES.get(SOURCE_LANG, {}).get("name_vi", "tự động phát hiện")
    tgt_name = LANGUAGES.get(TARGET_LANG, {}).get("name_vi", "tiếng Việt")
    
    # ... prompt logic unchanged ...
    lang_instruction = "Hãy tự động nhận dạng ngôn ngữ nguồn của văn bản trong ảnh."
    if translate:
        prompt = (
            f"Nhiệm vụ: {lang_instruction}\n"
            f"Hãy đọc toàn bộ văn bản có trong bức ảnh này (OCR) và dịch sang {tgt_name}.\n"
            "Định dạng kết quả trả về CHỈ gồm cấu trúc sau, tuyệt đối không thêm bất kỳ văn bản giải thích, lời thoại hay ký tự thừa nào khác:\n"
            "[ORIGINAL]\n"
            "<Nội dung gốc được nhận diện>\n"
            "[TRANSLATED]\n"
            "<Nội dung dịch tương ứng>"
        )
    else:
        prompt = (
            f"Nhiệm vụ: {lang_instruction}\n"
            "Hãy đọc và trích xuất toàn bộ văn bản có trong bức ảnh này (OCR).\n"
            "Lưu ý: Chỉ trả về nội dung văn bản gốc nhận diện được, tuyệt đối không dịch sang ngôn ngữ khác, không thêm bất kỳ văn bản giải thích hay ký tự thừa nào."
        )
    async with gemini_lock:
        ocr_chat = None
        try:
            # Tạo một phiên chat mới cho mỗi lần OCR để tránh bị ảnh hưởng bởi lịch sử chat trước đó
            ocr_chat = gemini_client.start_chat()
            response = await ocr_chat.send_message(prompt, files=[image_path])
            result_text = response.text.strip()
            if DELETE_GEMINI_HISTORY and ocr_chat.cid:
                try:
                    await gemini_client.delete_chat(ocr_chat.cid)
                except Exception as delete_err:
                    print(f"[Gemini OCR] Lỗi xóa history chat: {delete_err}")
            return result_text
        except Exception as e:
            print(f"[Gemini OCR Error] {e}")
            if DELETE_GEMINI_HISTORY and ocr_chat and ocr_chat.cid:
                try:
                    await gemini_client.delete_chat(ocr_chat.cid)
                except Exception as delete_err:
                    print(f"[Gemini OCR] Lỗi xóa history chat sau lỗi: {delete_err}")
            return f"ERROR: Lỗi gửi ảnh tới Gemini: {str(e)}"

def ocr_with_gemini_web(image_path: str, translate: bool = True) -> str | None:
    """Nhận diện chữ (OCR) đồng bộ bằng cách chạy coroutine trên loop nền."""
    if not gemini_chat:
        return "ERROR: Chưa kết nối Gemini Web. Vui lòng kết nối Profile ở nút hình người 👤."
    future = asyncio.run_coroutine_threadsafe(async_ocr_with_gemini_web(image_path, translate), gemini_loop)
    try:
        return future.result(timeout=30)
    except Exception as e:
        print(f"[Gemini OCR Timeout/Error] {e}")
        return f"ERROR: Hết thời gian chờ phản hồi từ Gemini: {str(e)}"

def build_gemini_system_prompt(ref_image: bool = True, template_mode: str = "default", keep_hair: bool = True, keep_acc: bool = False, keep_bg: bool = True, model_file: str = "None", user_suggestions: str = "") -> str:
    """Xây dựng chuỗi prompt hệ thống gửi đi dựa trên cấu hình."""
    # Try to load selected model json file if active
    mau_context = ""
    if ref_image and model_file != "None":
        mau_path = BASE_DIR / "mau" / model_file
        if mau_path.exists():
            try:
                with open(mau_path, "r", encoding="utf-8") as f:
                    mau_data = json.load(f)
                subject_desc = mau_data.get("subject", {}).get("description", "")
                hair_desc = str(mau_data.get("hair", ""))
                body_desc = str(mau_data.get("body", ""))
                # If keep_acc is False (loại bỏ phụ kiện), do not load original accessories info, or tell prompt to skip it.
                acc_desc = ""
                if not keep_acc:
                    acc_desc = "- Phụ kiện: Không giữ lại phụ kiện trên người (no accessories, no jewelry, no watches, no glasses, no headbands, no hats)\n"
                else:
                    acc_desc = f"- Phụ kiện: {str(mau_data.get('accessories', {}))}\n"
                
                 
                mau_context = (
                    f"Thông tin mẫu cần giữ nguyên:\n"
                    f"- Mô tả mẫu: {subject_desc}\n"
                    f"- Kiểu tóc: {hair_desc}\n"
                    f"- Vóc dáng/Cơ thể: {body_desc}\n"
                    f"{acc_desc}\n"
                )
            except:
                pass

    suggestion_instruction = ""
    if user_suggestions:
        suggestion_instruction = (
            f"QUAN TRỌNG: Hãy thiết lập bối cảnh (background), tư thế (pose), trang phục (clothing) hoặc phong cách chụp "
            f"dựa trên các gợi ý chi tiết sau của người dùng: \"{user_suggestions}\". Hãy đảm bảo thể hiện chính xác gợi ý này.\n"
        )

    quality_instruction = (
        "QUAN TRỌNG: Trong prompt sinh ra (hoặc các thuộc tính tương ứng trong JSON như 'photography.camera_style', 'body.skin.texture', 'constraints.avoid', 'negative_prompt'), hãy bắt buộc bao gồm các yêu cầu chất lượng sau:\n"
        "- Mô tả vật thể & Bối cảnh: Hãy quan sát và mô tả chính xác 100% từng chi tiết các vật thể, đồ đạc, phụ kiện, bối cảnh (objects, elements, furniture, props, overall scene layout) xuất hiện trong ảnh gốc, không bỏ sót và không thêm bớt chi tiết thừa.\n"
        "- Phân tích bối cảnh & Ánh sáng: Hãy mô tả bối cảnh (background/environment) càng chi tiết càng tốt, bao gồm màu sắc chủ đạo (colors), hướng và nguồn ánh sáng (lighting source/direction/colors), góc máy quay và cự ly chụp (camera angle, camera shot style) từ ảnh gốc để mô tả lại chân thực nhất.\n"
        "- Phong cách ảnh thật: 'RAW photo, candid photography, natural skin texture, unretouched skin, realistic pores, high dynamic range, photorealistic, authentic indoor lifestyle photography'.\n"
        "- Thiết bị chụp: 'Shot on iPhone 15 Pro Max, 48MP main camera, f/1.78 aperture, natural HDR, real smartphone photography'.\n"
        "- Đặc điểm da/mặt: Kết cấu da tự nhiên có lỗ chân lông, không làm mịn da quá mức (no excessive skin smoothing, natural skin texture with pores), không bóp méo khuôn mặt (no facial distortion), no visible seams, no cut borders, no uneven skin tone, no mismatched skin tone areas.\n"
        "- Phân tích Vóc dáng/Cơ thể & Da (Body & Skin): Phải quan sát và mô tả chính xác theo hình ảnh mới được tải lên, bao gồm dáng người (frame/build), các vùng cơ thể hiển thị (waist, chest, legs, skin visible areas), tông màu da thực tế (skin tone) và kết cấu da (skin texture), không rập khuôn hay sao chép từ mẫu cũ.\n"
        f"- Bối cảnh, tư thế, vóc dáng/cơ thể, trang phục, phụ kiện và phong cách chụp (background, pose, body, clothing, accessories, photography): Phải phân tích và mô tả chính xác 100% dựa theo hình ảnh mới được tải lên (hoặc theo gợi ý chi tiết của người dùng nếu có). \n"
    )

    if template_mode == "json":
        
        if keep_bg:
            bg_setting = "<Bối cảnh/Không gian>, from the uploaded reference image"
            bg_wall_color = "<Màu tường/Trần/Phía sau>, from the uploaded reference image"
            bg_elements = '[\n      "<Thành phần bối cảnh 1>, from the uploaded reference image",\n      "<Thành phần bối cảnh 2>, from the uploaded reference image"\n    ]'
            bg_atmosphere = "<Bầu không khí bối cảnh>, from the uploaded reference image"
            bg_lighting = "<Ánh sáng bối cảnh>, from the uploaded reference image"
        else:
            bg_setting = "<Bối cảnh/Không gian> ,  outdoor environment matching the uploaded reference image"
            bg_wall_color = "<Màu tường/Trần/Phía sau>"
            bg_elements = '[\n      "<Thành phần bối cảnh 1>",\n      "<Thành phần bối cảnh 2>"\n    ]'
            bg_atmosphere = "<Bầu không khí bối cảnh>"
            bg_lighting = "<Ánh sáng bối cảnh>"

        ref_instruction = (
            "Trong phần 'subject.description', hãy bắt buộc bắt đầu bằng: \"A high-fidelity portrait of the same woman from the uploaded reference image, a young Vietnamese woman from An Giang in the Mekong Delta, with authentic Southern Vietnamese features, soft see-through bangs, warm smile, natural complexion \". "
            "Khi mô tả chi tiết khuôn mặt, hãy chú ý dùng các từ khóa nhận dạng: oval face shape, soft V-shaped jawline, medium almond-shaped brown eyes, straight eyebrows with gentle arch, small straight nose, medium-full lips, fair East Asian skin tone, delicate facial features. "
            "Đồng thời, bắt buộc phải trả về thêm trường 'identity_preservation' nằm ngay sau 'subject' với giá trị cấu trúc chính xác như sau:\n"
            "  \"identity_preservation\": {\n"
            "    \"priority\": \"maximum\",\n"
            "    \"instructions\": [\n"
            "      \"Use the uploaded reference image as the primary facial reference\",\n"
            "      \"Preserve the exact facial identity from the reference image\",\n"
            "      \"Maintain original facial proportions\",\n"
            "      \"Maintain original eye shape and spacing\",\n"
            "      \"Maintain original nose shape\",\n"
            "      \"Maintain original lip shape\",\n"
            "      \"Maintain original jawline and chin shape\",\n"
            "      \"Do not beautify, modify, or reinterpret facial features\",\n"
            "      \"Keep eyes open\"\n"
            "    ]\n"
            "  }\n"
            "QUAN TRỌNG VỀ ĐỊNH DẠNG ẢNH SINH RA:\n"
            "- Nếu ảnh gốc/ảnh tham chiếu là một ảnh ghép nhiều tấm (grid, model sheet, character sheet, expression sheet, multi-panel, collage, split screen, multiple views) như ảnh mẫu biểu cảm, hãy dùng nó để tham chiếu khuôn mặt/cơ thể của nhân vật. Nhưng TUYỆT ĐỐI KHÔNG ĐƯỢC sinh ra mô tả ảnh ghép/nhiều ô. Hãy bắt buộc mô tả một bức ảnh duy nhất (single image, individual photo, single portrait) của nhân vật đó.\n"
            "- Trong trường 'constraints.avoid' và 'negative_prompt', hãy bắt buộc thêm vào các từ khóa sau để tránh tạo ảnh ghép và tránh xuất hiện chữ/văn bản/logo trên hình: \"multi-panel, grid, collage, split screen, multiple views, expression sheet, character sheet, model sheet, text, font, logo, signature, watermark, label, writing, letters, words\".\n"
        ) if ref_image else ""
        
        prompt = (
            "Hãy đóng vai chuyên gia Prompt Engineering. Dựa trên hình ảnh tham chiếu, hãy phân tích cực kỳ chi tiết bức ảnh và trả về một đối tượng JSON mô tả bức ảnh theo cấu trúc chính xác dưới đây.\n"
            f"{mau_context}"
            f"{suggestion_instruction}"
            f"{ref_instruction}\n"
            f"{quality_instruction}\n"
            "Yêu cầu: Viết nội dung phân tích chi tiết bằng tiếng Anh (hoặc dịch đúng các phần cần thiết). Trả về CHỈ duy nhất chuỗi JSON hợp lệ, tuyệt đối không được có markdown block (như ```json) hay văn bản giải thích nào khác ngoài chuỗi JSON.\n\n"
            "Cấu trúc JSON yêu cầu:\n"
            "{\n"
            "  \"subject\": {\n"
            "    \"description\": \"<Mô tả chi tiết chủ thể bằng tiếng Anh>\",\n"
            "    \"mirror_rules\": \"None.\",\n"
            "    \"age\": \"<Tuổi chủ thể>\",\n"
            "    \"expression\": {\n"
            "      \"eyes\": {\n"
            "        \"look\": \"<Mô tả ánh mắt>\",\n"
            "        \"energy\": \"<Năng lượng ánh mắt>\",\n"
            "        \"direction\": \"<Hướng nhìn của mắt>\"\n"
            "      },\n"
            "      \"mouth\": {\n"
            "        \"position\": \"<Vị trí/trạng thái miệng>\",\n"
            "        \"energy\": \"<Trạng thái của miệng>\"\n"
            "      },\n"
            "      \"overall\": \"<Biểu cảm tổng thể khuôn mặt>\"\n"
            "    },\n"
            "    \"face\": {\n"
            "      \"preserve_original\": \"True\",\n"
            "      \"makeup\": \"<Kiểu trang điểm>\"\n"
            "    },\n"
            "    \"facial_features\": {\n"
            "      \"face_shape\": \"Oval face with soft V-shaped jawline\",\n"
            "      \"eyes\": \"Medium almond-shaped brown eyes\",\n"
            "      \"eyebrows\": \"Straight eyebrows with a gentle arch\",\n"
            "      \"nose\": \"Small straight nose\",\n"
            "      \"lips\": \"Medium-full natural lips\",\n"
            "      \"skin\": \"Fair East Asian skin tone\"\n"
            "    }\n"
            "  },\n"
            "  \"identity_preservation\": {\n"
            "    \"priority\": \"maximum\",\n"
            "    \"instructions\": [\n"
            "      \"Use the uploaded reference image as the primary facial reference\",\n"
            "      \"Preserve the exact facial identity from the reference image\",\n"
            "      \"Maintain original facial proportions\",\n"
            "      \"Maintain original eye shape and spacing\",\n"
            "      \"Maintain original nose shape\",\n"
            "      \"Maintain original lip shape\",\n"
            "      \"Maintain original jawline and chin shape\",\n"
            "      \"Do not beautify, modify, or reinterpret facial features\",\n"
            "      \"Keep eyes open\"\n"
            "    ]\n"
            "  },\n"
            "  \"hair\": {\n"
            "    \"color\": \"<Màu tóc>\",\n"
            "    \"style\": \"<Kiểu tóc>\",\n"
            "    \"effect\": \"<Hiệu ứng tóc>\"\n"
            "  },\n"
            "  \"body\": {\n"
            "    \"frame\": \"<Khung người/Dáng người>\",\n"
            "    \"waist\": \"<Mô tả eo>\",\n"
            "    \"chest\": \"<Mô tả ngực>\",\n"
            "    \"legs\": \"<Mô tả chân>\",\n"
            "    \"skin\": {\n"
            "      \"visible_areas\": \"<Vùng da hở/lộ>\",\n"
            "      \"tone\": \"<Màu da/Tông da>\",\n"
            "      \"texture\": \"<Kết cấu da>\",\n"
            "      \"lighting_effect\": \"<Hiệu ứng ánh sáng lên da>\"\n"
            "    }\n"
            "  },\n"
            "  \"pose\": {\n"
            "    \"position\": \"<Vị trí tư thế>\",\n"
            "    \"base\": \"<Trọng tâm/Tư thế cơ sở>\",\n"
            "    \"overall\": \"<Mô tả tư thế tổng thể>\"\n"
            "  },\n"
            "  \"clothing\": {\n"
            "    \"top\": {\n"
            "      \"type\": \"<Áo/Phần trên áo>\",\n"
            "      \"color\": \"<Màu áo>\",\n"
            "      \"details\": \"<Chi tiết áo>\",\n"
            "      \"effect\": \"<Hiệu ứng áo>\"\n"
            "    },\n"
            "    \"bottom\": {\n"
            "      \"type\": \"<Quần/váy/Phần dưới áo>\",\n"
            "      \"color\": \"<Màu quần/váy>\",\n"
            "      \"details\": \"<Chi tiết quần/váy>\"\n"
            "    }\n"
            "  },\n"
            "  \"accessories\": {\n"
            "    \"prop\": \"<Vật dụng/Phụ kiện đi kèm>\"\n"
            "  },\n"
            "  \"photography\": {\n"
            "    \"camera_style\": \"<Phong cách camera>\",\n"
            "    \"angle\": \"<Góc chụp>\",\n"
            "    \"shot_type\": \"<Loại khung hình>\",\n"
            "    \"aspect_ratio\": \"<Tỷ lệ khung hình>\",\n"
            "    \"texture\": \"<Chất lượng ảnh>\",\n"
            "    \"lighting\": \"<Ánh sáng nhiếp ảnh>\",\n"
            "    \"depth_of_field\": \"<Độ sâu trường ảnh>\"\n"
            "  },\n"
            f"  \"background\": {{\n"
            f"    \"setting\": \"{bg_setting}\",\n"
            f"    \"wall_color\": \"{bg_wall_color}\",\n"
            f"    \"elements\": {bg_elements},\n"
            f"    \"atmosphere\": \"{bg_atmosphere}\",\n"
            f"    \"lighting\": \"{bg_lighting}\"\n"
            f"  }},\n"
            "  \"the_vibe\": {\n"
            "    \"energy\": \"<Năng lượng bức ảnh>\",\n"
            "    \"mood\": \"<Tâm trạng>\",\n"
            "    \"aesthetic\": \"<Phong cách thẩm mỹ>\",\n"
            "    \"authenticity\": \"<Mức độ tự nhiên chân thực>\",\n"
            "    \"intimacy\": \"<Mức độ thân mật>\",\n"
            "    \"story\": \"<Câu chuyện giả định của ảnh>\",\n"
            "    \"caption_energy\": \"<Caption gợi ý>\"\n"
            "  },\n"
            "  \"constraints\": {\n"
            "    \"must_keep\": [\n"
            "      \"<Yếu tố bắt buộc phải giữ lại 1>\",\n"
            "      \"<Yếu tố bắt buộc phải giữ lại 2>\"\n"
            "    ],\n"
            "    \"avoid\": [\n"
            "      \"<Yếu tố cần tránh 1>\",\n"
            "      \"<Yếu tố cần tránh 2>\"\n"
            "    ]\n"
            "  },\n"
            "  \"negative_prompt\": [\n"
            "      \"<Negative prompt 1>\",\n"
            "      \"<Negative prompt 2>\"\n"
            "  ]\n"
            "}"
        )
    else:
        hair_instruction = "kiểu tóc (color, style, effect)" if keep_hair else ""
        hair_sep = ", " if keep_hair else ""
        bg_instruction = "Tuy nhiên, hãy giữ nguyên tư thế (pose) của nhân vật từ ảnh gốc, còn bối cảnh (background) hãy lấy từ bối cảnh của mẫu tham chiếu, trừ khi có gợi ý thay đổi cụ thể từ người dùng dưới đây." if keep_bg else "Tuy nhiên, hãy giữ nguyên tư thế (pose) và bối cảnh (background) của nhân vật từ ảnh gốc, trừ khi có gợi ý thay đổi cụ thể từ người dùng dưới đây."
        ref_instruction = (
            "Prompt tiếng Anh phải bắt đầu bằng chính xác cụm từ sau: \"Same person as reference photo. Strong facial identity preservation. Exact face shape, eyes, nose, lips, jawline, eyebrows. No facial modification. \" "
            "và Prompt tiếng Việt phải bắt đầu bằng chính xác cụm từ sau: \"(Sử dụng hình ảnh tham chiếu được tải lên làm tham chiếu nhận dạng chính. Giữ nguyên hoàn toàn khuôn mặt của mẫu tham chiếu) \". "
            f"QUAN TRỌNG: Hãy giữ nguyên các đặc điểm diện mạo của mẫu/chủ thể (như khuôn mặt, {hair_instruction}{hair_sep}màu da, tuổi tác, trang điểm) VÀ giữ nguyên trang phục (clothing), phụ kiện (accessories) cùng phong cách nhiếp ảnh (photography) từ hình ảnh tham chiếu để giữ nguyên mẫu (TRỪ KHI có gợi ý thay đổi từ người dùng bên dưới). "
            "Khi mô tả khuôn mặt, hãy chú ý bổ sung các đặc điểm nhận dạng nổi bật để tăng độ bám khuôn mặt: oval face shape, soft V-shaped jawline, medium almond-shaped brown eyes, straight eyebrows with gentle arch, small straight nose, medium-full lips, fair East Asian skin tone, delicate facial features. "
            f"{bg_instruction}\n"
            "QUAN TRỌNG VỀ ĐỊNH DẠNG ẢNH SINH RA: Nếu ảnh gốc/ảnh tham chiếu là một ảnh ghép nhiều tấm (grid, model sheet, character sheet, expression sheet, multi-panel, collage, split screen, multiple views) như ảnh mẫu biểu cảm, hãy phân tích khuôn mặt/cơ thể của nhân vật từ đó. Nhưng TUYỆT ĐỐI KHÔNG ĐƯỢC sinh ra mô tả ảnh ghép/nhiều ô. Hãy bắt buộc mô tả một bức ảnh duy nhất (single image, individual photo, single portrait) của nhân vật đó. Đồng thời, hãy tự động đưa các từ khóa tránh tạo ảnh ghép và tránh xuất hiện chữ/văn bản/logo (ví dụ: 'multi-panel, grid, collage, split screen, multiple views, expression sheet, character sheet, model sheet, text, font, logo, signature, watermark, label, writing, letters, words') vào negative prompt. "
        ) if ref_image else ""
        
        prompt = (
            "Hãy đóng vai chuyên gia Prompt Engineering. Dựa trên hình ảnh tham chiếu, nhiệm vụ của bạn là VIẾT MỘT ĐOẠN VĂN BẢN MÔ TẢ (Prompt) dùng để sinh ảnh (TUYỆT ĐỐI KHÔNG ĐƯỢC TẠO/VẼ ẢNH, CHỈ VIẾT VĂN BẢN). Hãy tạo một prompt AI image generation hoàn chỉnh bằng tiếng Anh (PROMPT EN), tối ưu cho Midjourney, ChatGPT Images, Gemini, Flux và SDXL. "
            "Yêu cầu: Chỉ mô tả bằng văn bản những gì thực sự nhìn thấy trong ảnh. Không suy đoán danh tính, tuổi, quốc tịch, địa điểm hoặc các chi tiết không xuất hiện trong ảnh. "
            "Phân tích đầy đủ: Subject, Face, Hair, Clothing, Accessories, Pose, Expression, Environment, Lighting, Composition, Camera Angle, Lens, Color Palette, Style, Quality. "
            "Sử dụng thuật ngữ nhiếp ảnh và điện ảnh chuyên nghiệp. Viết thành một prompt liền mạch, tự nhiên, chi tiết và giàu hình ảnh bằng tiếng Anh. "
            f"{suggestion_instruction}"
            f"{ref_instruction}"
            f"{quality_instruction}"
            "Kết thúc bằng các từ khóa chất lượng cao phù hợp với ảnh.\n"
            "Sau đó, hãy dịch chính xác prompt tiếng Anh đó sang tiếng Việt (PROMPT VI).\n"
            "Định dạng kết quả trả về CHỈ gồm cấu trúc sau, tuyệt đối không thêm bất kỳ văn bản giải thích, không tạo ảnh hay ký tự thừa nào khác:\n"
            "[PROMPT_EN]\n"
            "<nội dung prompt tiếng Anh>\n"
            "[PROMPT_VI]\n"
            "<nội dung bản dịch tiếng Việt>"
        )
    return prompt

async def async_analyze_prompt_with_gemini_web(image_path: str, ref_image: bool = True, template_mode: str = "default", keep_hair: bool = True, keep_acc: bool = False, keep_bg: bool = True, model_file: str = "None", user_suggestions: str = "") -> str | None:
    """Phân tích ảnh và tạo prompt AI qua Gemini Web."""
    global gemini_client, gemini_chat, gemini_lock, DELETE_GEMINI_HISTORY
    if not gemini_client:
        return "ERROR: Chưa kết nối Gemini Web. Vui lòng kết nối Profile ở nút hình người 👤."
    if gemini_lock is None:
        gemini_lock = asyncio.Lock()
    
    prompt = build_gemini_system_prompt(ref_image, template_mode, keep_hair, keep_acc, keep_bg, model_file, user_suggestions)
    
    async with gemini_lock:
        prompt_chat = None
        try:
            prompt_chat = gemini_client.start_chat()
            files = [image_path] if image_path else []
            response = await prompt_chat.send_message(prompt, files=files)
            result_text = response.text.strip()
            if DELETE_GEMINI_HISTORY and prompt_chat.cid:
                try:
                    await gemini_client.delete_chat(prompt_chat.cid)
                except Exception as delete_err:
                    print(f"[Gemini Prompt] Lỗi xóa history chat: {delete_err}")
            return result_text
        except Exception as e:
            print(f"[Gemini Prompt Error] {e}")
            if DELETE_GEMINI_HISTORY and prompt_chat and prompt_chat.cid:
                try:
                    await gemini_client.delete_chat(prompt_chat.cid)
                except Exception as delete_err:
                    print(f"[Gemini Prompt] Lỗi xóa history chat sau lỗi: {delete_err}")
            return f"ERROR: Lỗi gửi ảnh tới Gemini: {str(e)}"

def analyze_prompt_with_gemini_web(image_path: str, ref_image: bool = True, template_mode: str = "default", keep_hair: bool = True, keep_acc: bool = False, keep_bg: bool = True, model_file: str = "None", user_suggestions: str = "") -> str | None:
    """Phân tích ảnh đồng bộ bằng cách chạy coroutine trên loop nền."""
    if not gemini_chat:
        return "ERROR: Chưa kết nối Gemini Web. Vui lòng kết nối Profile ở nút hình người 👤."
    future = asyncio.run_coroutine_threadsafe(async_analyze_prompt_with_gemini_web(image_path, ref_image, template_mode, keep_hair, keep_acc, keep_bg, model_file, user_suggestions), gemini_loop)
    try:
        return future.result(timeout=60)
    except Exception as e:
        print(f"[Gemini Prompt Timeout/Error] {e}")
        return f"ERROR: Hết thời gian chờ phản hồi từ Gemini: {str(e)}"

def init_gemini_client(profile_name: str) -> tuple[bool, str]:
    """Kết nối đồng bộ tới Gemini Client từ luồng ngoài."""
    future = asyncio.run_coroutine_threadsafe(async_init_gemini_client(profile_name), gemini_loop)
    try:
        return future.result(timeout=35)
    except Exception as e:
        return False, f"Hết thời gian kết nối: {str(e)}"

def list_profiles() -> list[str]:
    """Trả về danh sách tên profile trong thư mục profiles."""
    profiles_dir = BASE_DIR / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    return [d.name for d in sorted(profiles_dir.iterdir()) if d.is_dir()] or ["default"]

def load_gemini_profile() -> str:
    """Đọc profile đang hoạt động từ file cấu hình."""
    settings = load_settings()
    return settings.get("active_profile", "")

def save_gemini_profile(profile_name: str):
    """Lưu profile đang hoạt động vào file cấu hình."""
    save_settings({"active_profile": profile_name})

def delete_gemini_profile_file():
    """Xóa cấu hình profile đang hoạt động."""
    try:
        settings = load_settings()
        if "active_profile" in settings:
            del settings["active_profile"]
            config_path = BASE_DIR / "settings.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Lỗi xóa active_profile khỏi settings.json: {e}")

def find_chrome() -> str:
    """Tìm đường dẫn chạy Chrome trên Windows."""
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "chrome"

def translate_with_deepseek_api(text: str, api_key: str) -> str | None:
    """Dịch văn bản siêu tốc bằng API DeepSeek."""
    import requests
    src_name = LANGUAGES.get(SOURCE_LANG, {}).get("name_vi", "tiếng Nhật")
    tgt_name = LANGUAGES.get(TARGET_LANG, {}).get("name_vi", "tiếng Việt")
    
    system_msg = f"You are a professional real-time meeting translator. Translate the given text from {src_name} to {tgt_name} naturally and contextually. Output ONLY the translated text. Do not explain, do not add comments, no extra characters."
    if CURRENT_CONTEXT_PROMPT:
        system_msg += f"\n\nHere is the meeting context and custom domain instructions to follow:\n{CURRENT_CONTEXT_PROMPT}"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": text}
        ],
        "temperature": 0.3,
        "max_tokens": 1000
    }
    try:
        response = requests.post("https://api.deepseek.com/chat/completions", headers=headers, json=payload, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        else:
            print(f"[DeepSeek API Error] Status: {response.status_code}, Body: {response.text}")
    except Exception as e:
        print(f"[DeepSeek API Request Failed] {e}")
    return None

def translate_with_gemini_api(text: str, api_key: str) -> str | None:
    """Dịch văn bản siêu tốc bằng API Gemini chính thức (AI Studio)."""
    import requests
    src_name = LANGUAGES.get(SOURCE_LANG, {}).get("name_vi", "tiếng Nhật")
    tgt_name = LANGUAGES.get(TARGET_LANG, {}).get("name_vi", "tiếng Việt")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {
        "Content-Type": "application/json"
    }
    
    system_msg = (
        f"You are a professional real-time meeting translator. "
        f"Translate the following text from {src_name} to {tgt_name} naturally, fluently and contextually. "
        f"Output ONLY the translated text. Do not add any introduction, explanations, notes, or extra characters."
    )
    if CURRENT_CONTEXT_PROMPT:
        system_msg += f"\n\nHere is the meeting context and custom domain instructions to follow:\n{CURRENT_CONTEXT_PROMPT}"
        
    prompt = (
        f"{system_msg}\n\n"
        f"Text to translate:\n{text}"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1000
        }
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            print(f"[Gemini API Error] Status: {response.status_code}, Body: {response.text}")
    except Exception as e:
        print(f"[Gemini API Request Failed] {e}")
    return None

def translate_text(text: str) -> str:
    """Hàm dịch văn bản đa năng: thử dùng DeepSeek/Gemini API trước, sau đó là Gemini Web, cuối cùng là Google Dịch."""
    if LANGUAGES.get(TARGET_LANG, {}).get("google") == "none":
        return ""
        
    # 1. Thử dùng DeepSeek API
    if DEEPSEEK_API_KEY:
        try:
            vi_text = translate_with_deepseek_api(text, DEEPSEEK_API_KEY)
            if vi_text:
                return vi_text
        except Exception as e:
            print(f"[DeepSeek API Translation Failed] {e}")

    # 2. Thử dùng Gemini API chính thức
    if GEMINI_API_KEY:
        try:
            vi_text = translate_with_gemini_api(text, GEMINI_API_KEY)
            if vi_text:
                return vi_text
        except Exception as e:
            print(f"[Gemini API Translation Failed] {e}")

    # 3. Thử dùng Gemini Web (nếu có profile kết nối)
    if USE_GEMINI and gemini_chat:
        try:
            vi_text = translate_with_gemini(text)
            if vi_text:
                return vi_text
        except Exception as e:
            print(f"[Gemini Web Translation Failed, falling back to Google] {e}")
            
    # Fallback to GoogleTranslator
    try:
        if translator:
            return translator.translate(text)
    except Exception as e:
        print(f"[Google Translation Failed] {e}")
    return text

# ===============================
# WHISPER LOCAL INITIALIZATION
# ===============================

model = None
model_loading_lock = threading.Lock()
currently_loading_name = None
CURRENT_LOADED_MODEL_NAME = ""

def load_local_model(name: str):
    global model, currently_loading_name, CURRENT_LOADED_MODEL_NAME
    
    with model_loading_lock:
        if model is not None and CURRENT_LOADED_MODEL_NAME == name:
            return model
            
        if currently_loading_name == name:
            import time
            while currently_loading_name == name and model is None:
                time.sleep(0.1)
            if model is not None:
                return model
        
        currently_loading_name = name
        model = None
        
        from faster_whisper import WhisperModel
        msg_start = f"Bắt đầu tải/khởi chạy model cục bộ ({name}) về thư mục 'models'... Vui lòng đợi trong giây lát."
        print(msg_start)
        subtitle_queue.put(f"[SYSTEM]\n{msg_start}")
        
        try:
            models_dir = BASE_DIR / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            local_m = WhisperModel(
                name,
                device="cpu",
                compute_type="int8",
                download_root=str(models_dir)
            )
            
            model = local_m
            CURRENT_LOADED_MODEL_NAME = name
            currently_loading_name = None
            
            msg_end = f"Đã tải và kích hoạt thành công model cục bộ ({name})!"
            print(msg_end)
            subtitle_queue.put(f"[SYSTEM]\n{msg_end}")
            return model
        except Exception as ex:
            currently_loading_name = None
            CURRENT_LOADED_MODEL_NAME = ""
            msg_err = f"Lỗi khi tải model cục bộ ({name}): {ex}"
            print(msg_err)
            subtitle_queue.put(f"[SYSTEM]\n{msg_err}")
            return None

# ===============================
# UTILITIES
# ===============================

def resample(audio, orig_sr, target_sr):
    """Resample 1D float32 numpy array using linear interpolation."""
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    num_target_samples = int(duration * target_sr)
    src_indices = np.linspace(0, len(audio) - 1, num_target_samples)
    resampled_audio = np.interp(src_indices, np.arange(len(audio)), audio)
    return resampled_audio.astype(np.float32)

# ===============================
# TRANSCRIBER INSTANCE
# ===============================
from transcriber import AudioTranscriber

transcriber = AudioTranscriber(
    languages_dict=LANGUAGES,
    load_local_model_fn=load_local_model,
    transcribe_with_gemini_web_fn=transcribe_with_gemini_web
)

# ===============================
# SPEECH RECOGNITION WORKER
# ===============================

def speech_worker():
    global model
    buffer = []
    
    # Các biến trạng thái để nhận diện khoảng lặng ngắt câu kiểu Phiên dịch viên
    is_speaking = False
    silence_samples = 0
    
    # Cấu hình ngưỡng ngắt câu thông minh kiểu Phiên dịch viên (Interpretation Mode)
    MIN_DURATION = 1.0       # Giây tối thiểu của một câu để dịch (tránh nhiễu) - Giảm từ 1.8s
    MAX_DURATION = 15.0      # Giây tối đa bắt buộc phải dịch
    SILENCE_TIMEOUT = 0.35    # Giây im lặng liên tục để coi là đã nói xong 1 câu - Giảm từ 0.5s
    
    MIN_SAMPLES = int(SAMPLE_RATE * MIN_DURATION)
    MAX_SAMPLES = int(SAMPLE_RATE * MAX_DURATION)
    
    last_interim_time = time.time()

    while True:
        chunk = audio_queue.get()
        buffer.append(chunk)

        # Tính âm lượng (RMS) của chunk hiện tại
        chunk_volume = np.sqrt(np.mean(chunk**2)) if len(chunk) > 0 else 0
        total_samples = sum(len(x) for x in buffer)

        if chunk_volume >= SILENCE_THRESHOLD:
            is_speaking = True
            silence_samples = 0
        else:
            if is_speaking:
                silence_samples += len(chunk)

        # Nếu chưa nói, giữ pre-roll 0.5s để không bị mất chữ đầu khi bắt đầu nói
        if not is_speaking:
            max_preroll = int(SAMPLE_RATE * 0.5)
            while total_samples > max_preroll and len(buffer) > 1:
                buffer.pop(0)
                total_samples = sum(len(x) for x in buffer)

        # Thực hiện dịch nháp (Interim) định kỳ
        now = time.time()
        interim_interval = 1.2 if WHISPER_MODE == "online" else 0.8
        is_local_ready = (WHISPER_MODE != "local" or model is not None)
        
        if (is_speaking and 
            (now - last_interim_time >= interim_interval) and 
            (total_samples >= SAMPLE_RATE * 1.0) and 
            (WHISPER_MODE != "gemini") and 
            is_local_ready):
            
            last_interim_time = now
            try:
                audio_draft = np.concatenate(buffer, axis=0).flatten()
                
                # Hàm dịch nháp chạy nền để không block luồng thu âm chính
                def run_draft(audio_data):
                    try:
                        draft_text, _ = transcriber.transcribe(
                            audio_data,
                            mode=WHISPER_MODE,
                            src_lang=SOURCE_LANG,
                            local_model_name=LOCAL_MODEL_NAME,
                            api_url=API_URL,
                            gemini_chat=gemini_chat,
                            sample_rate=SAMPLE_RATE
                        )
                        if draft_text and draft_text.strip():
                            vi_draft = ""
                            if translator and LANGUAGES.get(TARGET_LANG, {}).get("google") != "none":
                                try:
                                    vi_draft = translator.translate(draft_text)
                                except Exception:
                                    pass
                            # Gửi tín hiệu dịch nháp về giao diện hiển thị
                            subtitle_queue.put(f"[INTERIM]\n{draft_text}\n{vi_draft}")
                    except Exception as e:
                        print(f"[DEBUG] Lỗi chép lời nháp: {e}")
                
                threading.Thread(target=run_draft, args=(audio_draft,), daemon=True).start()
            except Exception as e:
                print(f"[DEBUG] Lỗi khởi động luồng dịch nháp: {e}")

        # Điều kiện gửi đi dịch câu hoàn chỉnh:
        # 1. Đã tích lũy đủ độ dài tối thiểu (MIN_SAMPLES)
        # 2. Người nói đã im lặng đủ lâu (SILENCE_TIMEOUT) HOẶC đạt giới hạn tối đa (MAX_SAMPLES)
        is_silence_timeout = silence_samples >= SAMPLE_RATE * SILENCE_TIMEOUT
        is_max_limit_reached = total_samples >= MAX_SAMPLES

        if total_samples >= MIN_SAMPLES and (is_silence_timeout or is_max_limit_reached):
            try:
                audio = np.concatenate(buffer, axis=0)
                audio = audio.flatten()
                buffer.clear()

                # Reset trạng thái câu mới
                is_speaking = False
                silence_samples = 0
                last_interim_time = time.time()

                # Tính âm lượng tổng thể đoạn âm thanh để lọc nhiễu môi trường
                volume = np.sqrt(np.mean(audio**2))
                print(f"[DEBUG] Gửi dịch câu hoàn chỉnh | Độ dài: {len(audio)/SAMPLE_RATE:.2f}s | RMS: {volume:.6f}")

                if volume < SILENCE_THRESHOLD:
                    continue

                if WHISPER_MODE == "gemini" and not gemini_chat:
                    subtitle_queue.put("[SYSTEM]\nChưa kết nối Gemini Web. Hãy chọn profile và kết nối.")
                    subtitle_queue.put("[SYSTEM]\nSTOP_LISTENING:Chưa kết nối Gemini Web")
                    continue

                try:
                    original_text, segments = transcriber.transcribe(
                        audio,
                        mode=WHISPER_MODE,
                        src_lang=SOURCE_LANG,
                        local_model_name=LOCAL_MODEL_NAME,
                        api_url=API_URL,
                        gemini_chat=gemini_chat,
                        sample_rate=SAMPLE_RATE
                    )
                    
                    if WHISPER_MODE == "local":
                        if segments:
                            combined_text = " ".join(seg.text.strip() for seg in segments)
                            print(f"[DEBUG] Nhận diện được (Local): \"{combined_text}\"")
                            if combined_text.strip():
                                vi_text = translate_text(combined_text)
                                subtitle_queue.put(f"{combined_text}\n{vi_text}")
                        continue

                    # For online / gemini:
                    if original_text and original_text.strip():
                        print(f"[DEBUG] Nhận diện được ({WHISPER_MODE.capitalize()}): \"{original_text}\"")
                        vi_text = translate_text(original_text)
                        subtitle_queue.put(f"{original_text}\n{vi_text}")
                    else:
                        print("[DEBUG] Không phát hiện giọng nói trong câu này.")
                except Exception as err:
                    err_msg = str(err)
                    print(f"[ERROR] Lỗi transcribe ({WHISPER_MODE}): {err_msg}")
                    if WHISPER_MODE == "online":
                        subtitle_queue.put(f"[SYSTEM]\nSERVER_ERROR:{err_msg}")
            except Exception as e:
                print(e)

# ===============================
# SYSTEM SETTINGS DIALOG
# ===============================

class FileDownloadThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str) # success, file_path_or_error

    def __init__(self, urls, dest_path, is_zip=False, zip_member=None):
        super().__init__()
        # Ensure urls is a list of candidate links
        self.urls = [urls] if isinstance(urls, str) else list(urls)
        self.dest_path = dest_path
        self.is_zip = is_zip
        self.zip_member = zip_member

    def run(self):
        import requests
        last_error = ""
        for url in self.urls:
            try:
                r = requests.get(url, stream=True, timeout=30)
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                
                if self.is_zip:
                    import io
                    import zipfile
                    zip_data = io.BytesIO()
                    downloaded = 0
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            zip_data.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                self.progress.emit(int(downloaded * 100 / total_size))
                    
                    zip_data.seek(0)
                    with zipfile.ZipFile(zip_data) as z:
                        for name in z.namelist():
                            if name.lower() == self.zip_member.lower() or name.lower().endswith(self.zip_member.lower()):
                                with open(self.dest_path, "wb") as f:
                                    f.write(z.read(name))
                                self.finished.emit(True, str(self.dest_path))
                                return
                    raise Exception(f"Không tìm thấy {self.zip_member} trong file zip.")
                else:
                    downloaded = 0
                    with open(self.dest_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total_size > 0:
                                    self.progress.emit(int(downloaded * 100 / total_size))
                    self.finished.emit(True, str(self.dest_path))
                    return
            except Exception as e:
                last_error = str(e)
                continue
        self.finished.emit(False, last_error or "Tất cả liên kết tải đều thất bại.")

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_settings = load_settings()
        self.setWindowTitle("Cấu hình hệ thống")
        self.setMinimumSize(680, 480)

        
        from PyQt6.QtWidgets import QStackedWidget, QListWidget, QFrame, QHBoxLayout, QVBoxLayout, QWidget, QFormLayout
        
        # Horizontal Split layout
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Left Sidebar Panel
        sidebar_frame = QFrame()
        sidebar_frame.setObjectName("sidebar_frame")
        sidebar_layout = QVBoxLayout(sidebar_frame)
        sidebar_layout.setContentsMargins(10, 20, 10, 20)
        sidebar_layout.setSpacing(15)
        
        # Title in sidebar
        sidebar_title = QLabel("Cài đặt")
        sidebar_title.setObjectName("sidebar_title")
        sidebar_layout.addWidget(sidebar_title)
        
        # Sidebar Menu
        self.sidebar_menu = QListWidget()
        self.sidebar_menu.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.sidebar_menu.addItem("Dịch thuật")
        self.sidebar_menu.addItem("Giao diện")
        self.sidebar_menu.addItem("Tài khoản AI")
        self.sidebar_menu.addItem("Âm thanh")
        self.sidebar_menu.addItem("Hệ thống")
        self.sidebar_menu.setFixedWidth(160)
        sidebar_layout.addWidget(self.sidebar_menu)
        
        main_layout.addWidget(sidebar_frame)
        
        # Right Stacked Widget
        self.pages_widget = QStackedWidget()
        self.pages_widget.setObjectName("pages_widget")
        
        # Page 1: Translation
        page1 = QWidget()
        page1_layout = QVBoxLayout(page1)
        page1_layout.setContentsMargins(25, 25, 25, 25)
        page1_layout.setSpacing(15)
        
        page1_title = QLabel("Dịch thuật & Dữ liệu")
        page1_title.setObjectName("page_title")
        page1_layout.addWidget(page1_title)
        
        form1 = QFormLayout()
        form1.setSpacing(12)
        
        self.api_url_input = QLineEdit()
        self.api_url_input.setText(API_URL)
        form1.addRow(QLabel("API URL (STT Online):"), self.api_url_input)
        
        self.source_lang_select = QComboBox()
        self.source_lang_select.addItems(list(LANGUAGES.keys()))
        self.source_lang_select.setCurrentText(SOURCE_LANG)
        form1.addRow(QLabel("Ngôn ngữ gốc:"), self.source_lang_select)
        
        self.target_lang_select = QComboBox()
        self.target_lang_select.addItems([k for k in LANGUAGES.keys() if k != "Tự động (Auto)"])
        self.target_lang_select.setCurrentText(TARGET_LANG)
        form1.addRow(QLabel("Ngôn ngữ dịch:"), self.target_lang_select)

        # Công cụ chép lời STT
        self.model_select = QComboBox()
        self.model_select.addItem("Online")
        self.model_select.addItem("Gemini Web")
        self.model_select.addItem("Local (tiny)")
        self.model_select.addItem("Local (base)")
        self.model_select.addItem("Local (small)")
        self.model_select.addItem("Local (medium)")
        
        # Thiết lập giá trị mặc định dựa vào WHISPER_MODE và LOCAL_MODEL_NAME
        if WHISPER_MODE == "online":
            self.model_select.setCurrentText("Online")
        elif WHISPER_MODE == "gemini":
            self.model_select.setCurrentText("Gemini Web")
        else:
            self.model_select.setCurrentText(f"Local ({LOCAL_MODEL_NAME})")
        form1.addRow(QLabel("Động cơ chép lời (STT):"), self.model_select)

        self.use_gemini_checkbox = QCheckBox("Dịch bằng Gemini Web (nếu kết nối)")
        self.use_gemini_checkbox.setChecked(USE_GEMINI)
        form1.addRow(QLabel("Gemini Web:"), self.use_gemini_checkbox)

        self.delete_gemini_history_checkbox = QCheckBox("Xóa lịch sử cuộc trò chuyện trên Gemini")
        self.delete_gemini_history_checkbox.setChecked(DELETE_GEMINI_HISTORY)
        form1.addRow(QLabel("Xóa lịch sử Gemini:"), self.delete_gemini_history_checkbox)
        
        page1_layout.addLayout(form1)
        page1_layout.addStretch()
        self.pages_widget.addWidget(page1)
        
        # Page 1.5: Interface (Giao diện)
        page_int = QWidget()
        page_int_layout = QVBoxLayout(page_int)
        page_int_layout.setContentsMargins(25, 25, 25, 25)
        page_int_layout.setSpacing(15)
        
        page_int_title = QLabel("Cấu hình Giao diện")
        page_int_title.setObjectName("page_title")
        page_int_layout.addWidget(page_int_title)
        
        form_int = QFormLayout()
        form_int.setSpacing(12)
        
        # 1. Chọn Theme
        self.theme_select = QComboBox()
        self.theme_select.addItems(["Tối (Slate Dark)", "Sáng (Gemini Light)"])
        self.theme_select.setCurrentText("Tối (Slate Dark)" if THEME_MODE == "dark" else "Sáng (Gemini Light)")
        form_int.addRow(QLabel("Chủ đề (Theme):"), self.theme_select)
        
        # 2. Font chữ Ngôn ngữ gốc
        self.font_src_select = QComboBox()
        self.font_src_select.addItems([
            "'Segoe UI', sans-serif", "Arial", "Helvetica", "Tahoma", 
            "MS Gothic", "Yu Gothic", "Meiryo", "Times New Roman", "Courier New"
        ])
        self.font_src_select.setCurrentText(FONT_FAMILY_SRC)
        form_int.addRow(QLabel("Phông chữ gốc:"), self.font_src_select)
        
        # 3. Cỡ chữ Ngôn ngữ gốc
        self.size_src_select = QComboBox()
        self.size_src_select.addItems([str(x) for x in [12, 13, 14, 15, 16, 17, 18, 20, 22, 24, 26, 28]])
        self.size_src_select.setCurrentText(str(FONT_SIZE_SRC))
        form_int.addRow(QLabel("Cỡ chữ gốc (px):"), self.size_src_select)
        
        # 4. Font chữ Ngôn ngữ dịch
        self.font_tgt_select = QComboBox()
        self.font_tgt_select.addItems([
            "'Segoe UI', sans-serif", "Arial", "Helvetica", "Tahoma", 
            "Times New Roman", "Courier New"
        ])
        self.font_tgt_select.setCurrentText(FONT_FAMILY_TGT)
        form_int.addRow(QLabel("Phông chữ dịch:"), self.font_tgt_select)
        
        # 5. Cỡ chữ Ngôn ngữ dịch
        self.size_tgt_select = QComboBox()
        self.size_tgt_select.addItems([str(x) for x in [12, 13, 14, 15, 16, 17, 18, 20, 22, 24, 26, 28, 30, 32]])
        self.size_tgt_select.setCurrentText(str(FONT_SIZE_TGT))
        form_int.addRow(QLabel("Cỡ chữ dịch (px):"), self.size_tgt_select)
        
        page_int_layout.addLayout(form_int)
        page_int_layout.addStretch()
        self.pages_widget.addWidget(page_int)
        
        # Page 2: AI Accounts
        page2 = QWidget()
        page2_layout = QVBoxLayout(page2)
        page2_layout.setContentsMargins(25, 25, 25, 25)
        page2_layout.setSpacing(15)
        
        page2_title = QLabel("Tài khoản AI Dịch thuật")
        page2_title.setObjectName("page_title")
        page2_layout.addWidget(page2_title)
        
        form2 = QFormLayout()
        form2.setSpacing(12)
        
        self.deepseek_key_input = QLineEdit()
        self.deepseek_key_input.setText(DEEPSEEK_API_KEY)
        self.deepseek_key_input.setPlaceholderText("Nhập API Key DeepSeek để dịch siêu tốc")
        form2.addRow(QLabel("DeepSeek API Key:"), self.deepseek_key_input)
        
        self.gemini_key_input = QLineEdit()
        self.gemini_key_input.setText(GEMINI_API_KEY)
        self.gemini_key_input.setPlaceholderText("Nhập API Key Gemini từ AI Studio")
        form2.addRow(QLabel("Gemini API Key (AI Studio):"), self.gemini_key_input)
        
        page2_layout.addLayout(form2)
        page2_layout.addStretch()
        self.pages_widget.addWidget(page2)
        
        # Page 3: Audio Setup
        page3 = QWidget()
        page3_layout = QVBoxLayout(page3)
        page3_layout.setContentsMargins(25, 25, 25, 25)
        page3_layout.setSpacing(15)
        
        page3_title = QLabel("Cấu hình Thu âm")
        page3_title.setObjectName("page_title")
        page3_layout.addWidget(page3_title)
        
        form3 = QFormLayout()
        form3.setSpacing(12)
        
        self.record_mode_select = QComboBox()
        self.record_mode_select.addItems(["Loa (Loopback)", "Microphone"])
        self.record_mode_select.setCurrentText("Loa (Loopback)" if RECORD_MODE == "loopback" else "Microphone")
        form3.addRow(QLabel("Nguồn thu âm:"), self.record_mode_select)

        self.sample_rate_select = QComboBox()
        self.sample_rate_select.addItems(["16000", "32000", "44100", "48000"])
        self.sample_rate_select.setCurrentText(str(SAMPLE_RATE))
        form3.addRow(QLabel("Sample Rate (Hz):"), self.sample_rate_select)

        self.block_seconds_select = QComboBox()
        self.block_seconds_select.addItems(["2", "3", "4", "5", "6", "8", "10"])
        self.block_seconds_select.setCurrentText(str(BLOCK_SECONDS))
        form3.addRow(QLabel("Thời gian gom mẫu (giây):"), self.block_seconds_select)

        self.silence_threshold_select = QComboBox()
        self.silence_threshold_select.addItems(["0.001", "0.003", "0.005", "0.01", "0.02", "0.05"])
        self.silence_threshold_select.setCurrentText(str(SILENCE_THRESHOLD))
        form3.addRow(QLabel("Ngưỡng lọc im lặng (RMS):"), self.silence_threshold_select)
        
        page3_layout.addLayout(form3)
        page3_layout.addStretch()
        self.pages_widget.addWidget(page3)
        
        # Page 4: System Tools
        page4 = QWidget()
        page4_layout = QVBoxLayout(page4)
        page4_layout.setContentsMargins(25, 25, 25, 25)
        page4_layout.setSpacing(15)
        
        page4_title = QLabel("Tiện ích Hệ thống")
        page4_title.setObjectName("page_title")
        page4_layout.addWidget(page4_title)
        
        form4 = QFormLayout()
        form4.setSpacing(12)
        
        self.ffmpeg_path_input = QLineEdit()
        self.ffmpeg_path_input.setText(self.current_settings.get("ffmpeg_path", ""))
        self.ffmpeg_path_input.setPlaceholderText("Đường dẫn đến ffmpeg.exe hoặc thư mục chứa nó")
        
        ffmpeg_layout = QHBoxLayout()
        ffmpeg_layout.addWidget(self.ffmpeg_path_input)
        self.ffmpeg_browse_btn = QPushButton("Browse")
        self.ffmpeg_browse_btn.clicked.connect(self.browse_ffmpeg)
        ffmpeg_layout.addWidget(self.ffmpeg_browse_btn)
        self.ffmpeg_download_btn = QPushButton("Tải về")
        self.ffmpeg_download_btn.clicked.connect(self.download_ffmpeg)
        ffmpeg_layout.addWidget(self.ffmpeg_download_btn)
        form4.addRow(QLabel("Đường dẫn FFMPEG:"), ffmpeg_layout)
        
        self.yt_dlp_path_input = QLineEdit()
        self.yt_dlp_path_input.setText(self.current_settings.get("yt_dlp_path", ""))
        self.yt_dlp_path_input.setPlaceholderText("Đường dẫn đến file yt-dlp.exe")
        
        yt_dlp_layout = QHBoxLayout()
        yt_dlp_layout.addWidget(self.yt_dlp_path_input)
        self.yt_dlp_browse_btn = QPushButton("Browse")
        self.yt_dlp_browse_btn.clicked.connect(self.browse_yt_dlp)
        yt_dlp_layout.addWidget(self.yt_dlp_browse_btn)
        self.yt_dlp_download_btn = QPushButton("Tải về")
        self.yt_dlp_download_btn.clicked.connect(self.download_yt_dlp)
        yt_dlp_layout.addWidget(self.yt_dlp_download_btn)
        form4.addRow(QLabel("Đường dẫn yt-dlp:"), yt_dlp_layout)
        
        page4_layout.addLayout(form4)
        page4_layout.addStretch()
        self.pages_widget.addWidget(page4)
        
        main_layout.addWidget(self.pages_widget, 1)
        
        # Connect sidebar select changes to StackedWidget
        self.sidebar_menu.currentRowChanged.connect(self.pages_widget.setCurrentIndex)
        self.sidebar_menu.setCurrentRow(0)
        
        # Main Vertical layout for QDialog
        dialog_layout = QVBoxLayout(self)
        dialog_layout.setContentsMargins(0, 0, 0, 0)
        dialog_layout.setSpacing(0)
        dialog_layout.addLayout(main_layout, 1)
        
        # Separator line
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("settings_separator")
        dialog_layout.addWidget(sep)
        
        # Bottom button layout
        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(20, 12, 20, 12)
        bottom_bar.addStretch()
        
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.save_settings_data)
        self.button_box.rejected.connect(self.reject)
        
        bottom_bar.addWidget(self.button_box)
        dialog_layout.addLayout(bottom_bar)
        
        # Apply CSS styling for Settings Dialog
        if THEME_MODE == "light":
            dialog_bg = "#ffffff"
            sidebar_bg = "#f0f4f9"
            sidebar_item_selected = "#e2eaf4"
            text_color = "#1F1F1F"
            input_bg = "#ffffff"
            input_border = "rgba(0, 0, 0, 0.15)"
            input_border_focus = "#0B57D0"
            label_color = "#5F6368"
        else:
            dialog_bg = "#131314"
            sidebar_bg = "#1e1e20"
            sidebar_item_selected = "#2b2d3a"
            text_color = "#e3e3e3"
            input_bg = "rgba(255, 255, 255, 0.04)"
            input_border = "rgba(255, 255, 255, 0.12)"
            input_border_focus = "#8ab4f8"
            label_color = "#999999"

        self.setStyleSheet(f"""
            QDialog {{
                background-color: {dialog_bg};
                color: {text_color};
            }}
            QFrame#sidebar_frame {{
                background-color: {sidebar_bg};
                border-right: 1px solid {input_border};
                border-top-left-radius: 12px;
                border-bottom-left-radius: 12px;
            }}
            QLabel#sidebar_title {{
                color: {text_color};
                font-size: 16px;
                font-weight: bold;
                font-family: 'Segoe UI', sans-serif;
                margin-left: 10px;
                margin-bottom: 10px;
            }}
            QStackedWidget#pages_widget {{
                background-color: {dialog_bg};
                border-top-right-radius: 12px;
                border-bottom-right-radius: 12px;
            }}
            QLabel#page_title {{
                color: {input_border_focus};
                font-size: 18px;
                font-weight: bold;
                font-family: 'Segoe UI', sans-serif;
                margin-bottom: 10px;
            }}
            QFrame#settings_separator {{
                background-color: {input_border};
                border: none;
                height: 1px;
            }}
            QLabel {{
                color: {text_color};
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
            }}
            QListWidget {{
                background-color: {sidebar_bg};
                border: none;
                padding: 10px;
                outline: 0;
            }}
            QListWidget::item {{
                color: {text_color};
                padding: 8px 12px;
                border-radius: 8px;
                margin-bottom: 4px;
                font-weight: bold;
                font-size: 13px;
                font-family: 'Segoe UI', sans-serif;
            }}
            QListWidget::item:hover {{
                background-color: {sidebar_item_selected}50;
                color: {input_border_focus};
            }}
            QListWidget::item:selected {{
                background-color: {sidebar_item_selected};
                color: {input_border_focus};
                border: none;
                outline: 0;
            }}
            QListWidget::item:selected:active {{
                background-color: {sidebar_item_selected};
                color: {input_border_focus};
            }}
            QListWidget::item:selected:!active {{
                background-color: {sidebar_item_selected};
                color: {input_border_focus};
            }}
            QLineEdit {{
                background-color: {input_bg};
                color: {text_color};
                border: 1px solid {input_border};
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 13px;
                font-family: 'Segoe UI', sans-serif;
            }}
            QLineEdit:focus {{
                border: 1.5px solid {input_border_focus};
                background-color: {dialog_bg};
            }}
            QComboBox {{
                background-color: {input_bg};
                color: {text_color};
                border: 1px solid {input_border};
                border-radius: 6px;
                padding: 6px 24px 6px 12px;
                font-size: 13px;
                font-family: 'Segoe UI', sans-serif;
            }}
            QComboBox:focus {{
                border: 1.5px solid {input_border_focus};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid {text_color};
                width: 0;
                height: 0;
                margin-right: 10px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {dialog_bg};
                color: {text_color};
                selection-background-color: {sidebar_item_selected};
                border: 1px solid {input_border};
                border-radius: 6px;
                outline: 0;
            }}
            QCheckBox {{
                color: {text_color};
                font-size: 13px;
                font-family: 'Segoe UI', sans-serif;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 1.5px solid {input_border};
                border-radius: 4px;
                background-color: {input_bg};
            }}
            QCheckBox::indicator:checked {{
                background-color: {input_border_focus};
                border-color: {input_border_focus};
            }}
            QPushButton {{
                background-color: {input_bg};
                color: {text_color};
                border: 1px solid {input_border};
                border-radius: 16px;
                padding: 8px 24px;
                font-weight: bold;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: {sidebar_item_selected}80;
                border-color: {input_border_focus};
            }}
        """)
        
        save_btn = self.button_box.button(QDialogButtonBox.StandardButton.Save)
        if save_btn:
            save_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {input_border_focus};
                    color: {'#131314' if THEME_MODE == 'light' else '#ffffff'};
                    border: 1px solid {input_border_focus};
                    border-radius: 16px;
                    padding: 8px 24px;
                    font-weight: bold;
                    font-size: 13px;
                }}
                QPushButton:hover {{
                    background-color: {'#1a73e8' if THEME_MODE == 'light' else '#a3c7ff'};
                    border-color: {'#1a73e8' if THEME_MODE == 'light' else '#a3c7ff'};
                }}
            """)
            save_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        cancel_btn = self.button_box.button(QDialogButtonBox.StandardButton.Cancel)
        if cancel_btn:
            cancel_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    color: {text_color};
                    border: 1px solid {input_border};
                    border-radius: 16px;
                    padding: 8px 24px;
                    font-weight: bold;
                    font-size: 13px;
                }}
                QPushButton:hover {{
                    background-color: rgba(255, 255, 255, 0.05);
                    border-color: {text_color};
                }}
            """)
            cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)

    def browse_ffmpeg(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn file FFMPEG (ffmpeg.exe)", "", "FFMPEG Executable (ffmpeg.exe ffmpeg);;All Files (*)")
        if file_path:
            self.ffmpeg_path_input.setText(file_path)

    def browse_yt_dlp(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn file yt-dlp (yt-dlp.exe)", "", "yt-dlp Executable (yt-dlp.exe yt-dlp);;All Files (*)")
        if file_path:
            self.yt_dlp_path_input.setText(file_path)

    def download_ffmpeg(self):
        dest_path = BASE_DIR / "ffmpeg.exe"
        urls = [
            "https://github.com/ffbinaries/ffbinaries-prebuilt/releases/download/v4.4.1/ffmpeg-4.4.1-win-64.zip",
            "https://node.ffbinaries.com/bin/windows-64/ffmpeg.zip"
        ]
        
        self.ffmpeg_download_btn.setEnabled(False)
        self.ffmpeg_download_btn.setText("0%")
        
        self.ffmpeg_thread = FileDownloadThread(urls, dest_path, is_zip=True, zip_member="ffmpeg.exe")
        self.ffmpeg_thread.progress.connect(lambda p: self.ffmpeg_download_btn.setText(f"{p}%"))
        
        def on_finished(success, result):
            self.ffmpeg_download_btn.setEnabled(True)
            self.ffmpeg_download_btn.setText("Tải về")
            if success:
                self.ffmpeg_path_input.setText(result)
                QMessageBox.information(self, "Thành công", "Đã tải xong ffmpeg.exe!")
                
                # Check for ffprobe.exe as well, download it in background if missing
                ffprobe_path = BASE_DIR / "ffprobe.exe"
                if not ffprobe_path.exists():
                    self.download_ffprobe_bg()
            else:
                QMessageBox.critical(self, "Thất bại", f"Không thể tải FFMPEG:\n{result}")
                
        self.ffmpeg_thread.finished.connect(on_finished)
        self.ffmpeg_thread.start()

    def download_ffprobe_bg(self):
        # Quietly download ffprobe in background since it's also needed
        dest_path = BASE_DIR / "ffprobe.exe"
        urls = [
            "https://github.com/ffbinaries/ffbinaries-prebuilt/releases/download/v4.4.1/ffprobe-4.4.1-win-64.zip",
            "https://node.ffbinaries.com/bin/windows-64/ffprobe.zip"
        ]
        self.ffprobe_thread = FileDownloadThread(urls, dest_path, is_zip=True, zip_member="ffprobe.exe")
        self.ffprobe_thread.start()

    def download_yt_dlp(self):
        dest_path = BASE_DIR / "yt-dlp.exe"
        url = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe"
        
        self.yt_dlp_download_btn.setEnabled(False)
        self.yt_dlp_download_btn.setText("0%")
        
        self.yt_dlp_thread = FileDownloadThread(url, dest_path, is_zip=False)
        self.yt_dlp_thread.progress.connect(lambda p: self.yt_dlp_download_btn.setText(f"{p}%"))
        
        def on_finished(success, result):
            self.yt_dlp_download_btn.setEnabled(True)
            self.yt_dlp_download_btn.setText("Tải về")
            if success:
                self.yt_dlp_path_input.setText(result)
                QMessageBox.information(self, "Thành công", "Đã tải xong yt-dlp.exe!")
            else:
                QMessageBox.critical(self, "Thất bại", f"Không thể tải yt-dlp:\n{result}")
                
        self.yt_dlp_thread.finished.connect(on_finished)
        self.yt_dlp_thread.start()
        
    def save_settings_data(self):
        global API_URL, SOURCE_LANG, TARGET_LANG, RECORD_MODE, SAMPLE_RATE, BLOCK_SECONDS, SILENCE_THRESHOLD, USE_GEMINI, FFMPEG_PATH, YT_DLP_PATH, DELETE_GEMINI_HISTORY, DEEPSEEK_API_KEY, GEMINI_API_KEY, THEME_MODE, FONT_FAMILY_SRC, FONT_FAMILY_TGT, FONT_SIZE_SRC, FONT_SIZE_TGT, WHISPER_MODE, LOCAL_MODEL_NAME
        url = self.api_url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Cảnh báo", "API URL không được để trống!")
            return
        API_URL = url
        SOURCE_LANG = self.source_lang_select.currentText()
        TARGET_LANG = self.target_lang_select.currentText()
        
        new_mode = "loopback" if self.record_mode_select.currentText() == "Loa (Loopback)" else "mic"
        new_rate = int(self.sample_rate_select.currentText())
        new_block = int(self.block_seconds_select.currentText())
        # Retrieve hotkey shortcuts from current settings (since tab interface is removed)
        shortcut_tab1 = self.current_settings.get("shortcut_tab1", "Ctrl+1")
        shortcut_tab2 = self.current_settings.get("shortcut_tab2", "Ctrl+2")
        shortcut_tab3 = self.current_settings.get("shortcut_tab3", "Ctrl+3")
        shortcut_tab4 = self.current_settings.get("shortcut_tab4", "Ctrl+4")
        shortcut_close_tab = self.current_settings.get("shortcut_close_tab", "Ctrl+W")
        shortcut_ocr = self.current_settings.get("shortcut_ocr", "Ctrl+Q")
        new_silence = float(self.silence_threshold_select.currentText())
        new_use_gemini = self.use_gemini_checkbox.isChecked()
        new_delete_gemini_history = self.delete_gemini_history_checkbox.isChecked()
        ffmpeg_path = self.ffmpeg_path_input.text().strip()
        yt_dlp_path = self.yt_dlp_path_input.text().strip()
        
        DEEPSEEK_API_KEY = self.deepseek_key_input.text().strip()
        GEMINI_API_KEY = self.gemini_key_input.text().strip()
        
        # Whisper mode selection
        selected_engine = self.model_select.currentText()
        if selected_engine == "Online":
            new_whisper_mode = "online"
            new_local_model = LOCAL_MODEL_NAME
        elif selected_engine == "Gemini Web":
            new_whisper_mode = "gemini"
            new_local_model = LOCAL_MODEL_NAME
        else:
            new_whisper_mode = "local"
            new_local_model = selected_engine.split("(")[1].split(")")[0]
            
        # Interface configurations
        new_theme = "dark" if "Tối" in self.theme_select.currentText() else "light"
        font_src = self.font_src_select.currentText()
        font_tgt = self.font_tgt_select.currentText()
        size_src = int(self.size_src_select.currentText())
        size_tgt = int(self.size_tgt_select.currentText())
        
        if RECORD_MODE != new_mode or SAMPLE_RATE != new_rate:
            QMessageBox.information(
                self, "Yêu cầu khởi động lại",
                "Bạn đã thay đổi Nguồn thu âm hoặc Sample Rate. Vui lòng khởi động lại ứng dụng để các thay đổi này có hiệu lực."
            )
            
        RECORD_MODE = new_mode
        SAMPLE_RATE = new_rate
        BLOCK_SECONDS = new_block
        SILENCE_THRESHOLD = new_silence
        USE_GEMINI = new_use_gemini
        DELETE_GEMINI_HISTORY = new_delete_gemini_history
        FFMPEG_PATH = ffmpeg_path
        YT_DLP_PATH = yt_dlp_path
        
        WHISPER_MODE = new_whisper_mode
        LOCAL_MODEL_NAME = new_local_model
        
        THEME_MODE = new_theme
        FONT_FAMILY_SRC = font_src
        FONT_FAMILY_TGT = font_tgt
        FONT_SIZE_SRC = size_src
        FONT_SIZE_TGT = size_tgt
        
        save_settings({
            "api_url": API_URL,
            "source_lang": SOURCE_LANG,
            "target_lang": TARGET_LANG,
            "record_mode": RECORD_MODE,
            "sample_rate": SAMPLE_RATE,
            "block_seconds": BLOCK_SECONDS,
            "silence_threshold": SILENCE_THRESHOLD,
            "use_gemini": USE_GEMINI,
            "ffmpeg_path": FFMPEG_PATH,
            "yt_dlp_path": YT_DLP_PATH,
            "shortcut_tab1": shortcut_tab1,
            "shortcut_tab2": shortcut_tab2,
            "shortcut_tab3": shortcut_tab3,
            "shortcut_tab4": shortcut_tab4,
            "shortcut_close_tab": shortcut_close_tab,
            "shortcut_ocr": shortcut_ocr,
            "delete_gemini_history": DELETE_GEMINI_HISTORY,
            "deepseek_api_key": DEEPSEEK_API_KEY,
            "gemini_api_key": GEMINI_API_KEY,
            "whisper_mode": WHISPER_MODE,
            "local_model_name": LOCAL_MODEL_NAME,
            "theme_mode": THEME_MODE,
            "font_family_src": FONT_FAMILY_SRC,
            "font_family_tgt": FONT_FAMILY_TGT,
            "font_size_src": FONT_SIZE_SRC,
            "font_size_tgt": FONT_SIZE_TGT
        })
        update_translation_config()
        self.accept()

# ===============================
# GEMINI PROFILE DIALOG
# ===============================

class GeminiProfileDialog(QDialog):
    def __init__(self, parent=None, active_profile=""):
        super().__init__(parent)
        self.setWindowTitle("Cấu hình Gemini Profiles")
        self.setMinimumWidth(450)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {DRACULA['background']};
                border: 1px solid {DRACULA['current_line']};
                border-radius: 12px;
            }}
            QLabel {{
                color: {DRACULA['foreground']};
                font-size: 12px;
                font-family: 'Segoe UI', sans-serif;
                font-weight: bold;
            }}
            QComboBox {{
                background-color: {DRACULA['current_line']};
                color: {DRACULA['foreground']};
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 12px;
                font-family: 'Segoe UI', sans-serif;
            }}
            QComboBox QAbstractItemView {{
                background-color: {DRACULA['background']};
                color: {DRACULA['foreground']};
                selection-background-color: {DRACULA['current_line']};
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 6px;
            }}
            QPushButton {{
                color: {DRACULA['foreground']};
                font-size: 12px;
                font-weight: bold;
                font-family: 'Segoe UI', sans-serif;
                background-color: {DRACULA['current_line']};
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 6px;
                padding: 6px 14px;
            }}
            QPushButton:hover {{
                background-color: {DRACULA['cyan']};
                color: {DRACULA['background']};
                border-color: {DRACULA['cyan']};
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Hàng chọn profile
        select_layout = QHBoxLayout()
        select_layout.addWidget(QLabel("Chọn Profile:"))
        self.profile_select = QComboBox()
        select_layout.addWidget(self.profile_select, 1)
        layout.addLayout(select_layout)
        
        # Hàng thao tác profile
        profile_actions_layout = QHBoxLayout()
        self.add_btn = QPushButton("➕ Tạo Profile mới")
        self.add_btn.clicked.connect(self.add_new_profile)
        profile_actions_layout.addWidget(self.add_btn)
        
        self.delete_btn = QPushButton("🗑️ Xóa Profile")
        self.delete_btn.setStyleSheet(f"""
            QPushButton {{
                color: {DRACULA['foreground']};
                background-color: transparent;
                border: 1px solid {DRACULA['btn_red']};
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {DRACULA['btn_red']};
                color: white;
            }}
        """)
        self.delete_btn.clicked.connect(self.delete_profile)
        profile_actions_layout.addWidget(self.delete_btn)
        layout.addLayout(profile_actions_layout)
        
        # Hàng thao tác đăng nhập Chrome
        chrome_layout = QHBoxLayout()
        self.open_chrome_btn = QPushButton("🌐 Mở Chrome đăng nhập Google")
        self.open_chrome_btn.setStyleSheet(f"""
            QPushButton {{
                color: {DRACULA['foreground']};
                background-color: transparent;
                border: 1px solid {DRACULA['cyan']};
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {DRACULA['cyan']};
                color: {DRACULA['background']};
            }}
        """)
        self.open_chrome_btn.clicked.connect(self.open_chrome_login)
        chrome_layout.addWidget(self.open_chrome_btn, 1)
        layout.addLayout(chrome_layout)
        
        # Trạng thái kết nối
        self.status_label = QLabel("Trạng thái: Chưa kiểm tra")
        self.status_label.setStyleSheet("color: #aaa; font-style: italic;")
        layout.addWidget(self.status_label)
        
        # Nút kiểm tra kết nối / đọc cookie
        self.check_btn = QPushButton("✅ Đăng nhập xong & Đọc Cookie")
        self.check_btn.setStyleSheet(f"""
            QPushButton {{
                color: {DRACULA['foreground']};
                background-color: transparent;
                border: 1px solid {DRACULA['btn_green']};
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {DRACULA['btn_green']};
                color: white;
            }}
        """)
        self.check_btn.clicked.connect(self.check_and_connect)
        layout.addWidget(self.check_btn)
        
        # Nút Lưu và Hủy ở cuối
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.save_and_accept)
        self.button_box.rejected.connect(self.reject)
        save_btn = self.button_box.button(QDialogButtonBox.StandardButton.Save)
        if save_btn:
            save_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {DRACULA['btn_green']};
                    color: white;
                    border: 1px solid {DRACULA['btn_green']};
                    border-radius: 15px;
                    padding: 7px 22px;
                    font-weight: bold;
                    font-size: 12px;
                }}
                QPushButton:hover {{
                    background-color: {DRACULA['btn_green_hover']};
                    border-color: {DRACULA['btn_green_hover']};
                }}
            """)
            save_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        cancel_btn = self.button_box.button(QDialogButtonBox.StandardButton.Cancel)
        if cancel_btn:
            cancel_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    color: {DRACULA['foreground']};
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    border-radius: 15px;
                    padding: 7px 22px;
                    font-weight: bold;
                    font-size: 12px;
                }}
                QPushButton:hover {{
                    background-color: {DRACULA['current_line']};
                    border-color: {DRACULA['cyan']};
                }}
            """)
            cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        layout.addWidget(self.button_box)
        
        # Khởi tạo danh sách
        self.refresh_profile_list(active_profile)
        
    def refresh_profile_list(self, select_name=""):
        self.profile_select.clear()
        self.profile_select.addItem("<Không sử dụng>")
        profiles = list_profiles()
        self.profile_select.addItems(profiles)
        if select_name and select_name in profiles:
            self.profile_select.setCurrentText(select_name)
        elif select_name == "":
            self.profile_select.setCurrentIndex(0)
        elif "default" in profiles:
            self.profile_select.setCurrentText("default")
            
    def get_selected_profile(self):
        return self.profile_select.currentText()
        
    def add_new_profile(self):
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Tạo Profile mới", "Nhập tên profile (không dấu cách):")
        if ok and name:
            name = name.strip().replace(" ", "_")
            profile_path = BASE_DIR / "profiles" / name
            profile_path.mkdir(parents=True, exist_ok=True)
            self.refresh_profile_list(name)
            self.status_label.setText(f"Đã tạo profile: {name}. Vui lòng mở Chrome để đăng nhập.")
            
    def delete_profile(self):
        profile_name = self.get_selected_profile()
        if not profile_name or profile_name == "<Không sử dụng>":
            return
        if profile_name == "default":
            QMessageBox.warning(self, "Cảnh báo", "Không thể xóa profile default!")
            return
            
        ret = QMessageBox.question(
            self, "Xác nhận xóa", 
            f"Bạn có chắc muốn xóa profile '{profile_name}' và toàn bộ dữ liệu Chrome của nó?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if ret == QMessageBox.StandardButton.Yes:
            profile_path = BASE_DIR / "profiles" / profile_name
            import shutil
            try:
                shutil.rmtree(profile_path, ignore_errors=True)
            except Exception as e:
                QMessageBox.warning(self, "Lỗi", f"Không thể xóa thư mục profile: {e}")
            self.refresh_profile_list("default")
            self.status_label.setText(f"Đã xóa profile '{profile_name}'")
            
    def open_chrome_login(self):
        profile_name = self.get_selected_profile()
        if not profile_name or profile_name == "<Không sử dụng>":
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn hoặc tạo một profile hợp lệ để mở Chrome!")
            return
        profile_path = BASE_DIR / "profiles" / profile_name
        profile_path.mkdir(parents=True, exist_ok=True)
        
        chrome_exe = find_chrome()
        self.status_label.setText("⏳ Đang mở Chrome...")
        
        # Mở Chrome độc lập với profile riêng
        try:
            subprocess.Popen([
                chrome_exe,
                f"--user-data-dir={profile_path}",
                "--no-first-run",
                "--no-default-browser-check",
                "https://gemini.google.com"
            ])
            self.status_label.setText("🌐 Đang đăng nhập Google trên Chrome. Đóng Chrome hoàn toàn khi xong!")
            self.status_label.setStyleSheet("color: #FBBC04; font-weight: bold;")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể mở Chrome: {e}")
            self.status_label.setText("❌ Không mở được Chrome!")
            
    def check_and_connect(self):
        profile_name = self.get_selected_profile()
        if not profile_name or profile_name == "<Không sử dụng>":
            self.status_label.setText("Vui lòng chọn profile để kiểm tra.")
            self.status_label.setStyleSheet("color: #aaa;")
            return
        self.status_label.setText("⏳ Đang kiểm tra cookie & kết nối...")
        self.status_label.setStyleSheet("color: #aaa;")
        
        # Gọi kết nối đồng bộ qua helper
        ok, msg = init_gemini_client(profile_name)
        if ok:
            self.status_label.setText("✅ Kết nối thành công!")
            self.status_label.setStyleSheet("color: #34A853; font-weight: bold;")
        else:
            self.status_label.setText(f"❌ {msg}")
            self.status_label.setStyleSheet("color: #EA4335; font-weight: bold;")
            
    def save_and_accept(self):
        profile_name = self.get_selected_profile()
        if profile_name == "<Không sử dụng>":
            delete_gemini_profile_file()
            global gemini_client, gemini_chat, active_gemini_profile
            gemini_client = None
            gemini_chat = None
            active_gemini_profile = None
        else:
            save_gemini_profile(profile_name)
        self.accept()

class ShortcutRecorderDialog(QDialog):
    def __init__(self, parent=None, current_shortcut=""):
        super().__init__(parent)
        self.setWindowTitle("Activation shortcut")
        self.setFixedSize(450, 320)
        self.current_shortcut = current_shortcut
        self.recorded_text = current_shortcut
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.init_ui()

    def init_ui(self):
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {DRACULA['background']};
                color: {DRACULA['foreground']};
            }}
            QLabel {{
                color: {DRACULA['foreground']};
                font-family: 'Segoe UI', sans-serif;
            }}
            QPushButton {{
                color: {DRACULA['foreground']};
                font-size: 12px;
                font-family: 'Segoe UI', sans-serif;
                background-color: {DRACULA['current_line']};
                border: 1px solid {DRACULA['comment']};
                border-radius: 6px;
                padding: 6px 12px;
            }}
            QPushButton:hover {{
                background-color: {DRACULA['purple']};
                color: {DRACULA['background']};
            }}
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Title & Subtitle
        title = QLabel("Activation shortcut")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        main_layout.addWidget(title)
        
        subtitle = QLabel("A shortcut should start with Windows key, Ctrl, Alt or Shift.")
        subtitle.setStyleSheet(f"font-size: 12px; color: rgba(248, 248, 242, 150);")
        subtitle.setWordWrap(True)
        main_layout.addWidget(subtitle)
        
        # Recording Box Frame
        self.recording_box = QFrame()
        self.recording_box.setFixedHeight(70)
        self.box_layout = QHBoxLayout(self.recording_box)
        self.box_layout.setSpacing(8)
        main_layout.addWidget(self.recording_box)
        
        # Reset & Clear row
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(15)
        
        self.reset_btn = QPushButton("🔄 Reset")
        self.reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reset_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {DRACULA['comment']};
                font-size: 12px;
            }}
            QPushButton:hover {{
                color: {DRACULA['cyan']};
            }}
        """)
        self.reset_btn.clicked.connect(self.reset_shortcut)
        actions_layout.addWidget(self.reset_btn)
        
        self.clear_btn = QPushButton("❌ Clear")
        self.clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {DRACULA['comment']};
                font-size: 12px;
            }}
            QPushButton:hover {{
                color: {DRACULA['cyan']};
            }}
        """)
        self.clear_btn.clicked.connect(self.clear_shortcut)
        actions_layout.addWidget(self.clear_btn)
        actions_layout.addStretch()
        main_layout.addLayout(actions_layout)
        
        # Warning bar
        self.warning_bar = QFrame()
        self.warning_bar.setStyleSheet(f"""
            QFrame {{
                background-color: rgba(255, 85, 85, 20);
                border: 1px solid rgba(255, 85, 85, 50);
                border-radius: 6px;
            }}
        """)
        warning_layout = QHBoxLayout(self.warning_bar)
        warning_layout.setContentsMargins(10, 6, 10, 6)
        
        self.warning_icon = QLabel("❌")
        self.warning_icon.setStyleSheet(f"font-size: 12px; color: {DRACULA['red']};")
        warning_layout.addWidget(self.warning_icon)
        
        self.warning_msg = QLabel("Invalid shortcut")
        self.warning_msg.setStyleSheet(f"font-size: 12px; color: {DRACULA['red']}; font-weight: 500;")
        warning_layout.addWidget(self.warning_msg)
        warning_layout.addStretch()
        
        main_layout.addWidget(self.warning_bar)
        main_layout.addStretch()
        
        # Bottom Save & Cancel buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self.save_btn = QPushButton("Save")
        self.save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {DRACULA['cyan']};
                color: black;
                font-weight: bold;
                border: none;
                border-radius: 6px;
                padding: 6px 20px;
            }}
            QPushButton:hover {{
                background-color: {DRACULA['purple']};
            }}
            QPushButton:disabled {{
                background-color: rgba(68, 71, 90, 50);
                color: rgba(248, 248, 242, 60);
                border: 1px solid rgba(98, 114, 164, 20);
            }}
        """)
        self.save_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.save_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)
        
        main_layout.addLayout(btn_layout)
        
        self.update_display()
        self.setFocus()

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
            
        key = event.key()
        modifiers = event.modifiers()
        is_mod = key in (Qt.Key.Key_Control, Qt.Key.Key_Shift, Qt.Key.Key_Alt, Qt.Key.Key_Meta)
        
        parts = []
        if modifiers & Qt.KeyboardModifier.ControlModifier or key == Qt.Key.Key_Control:
            parts.append("Ctrl")
        if modifiers & Qt.KeyboardModifier.AltModifier or key == Qt.Key.Key_Alt:
            parts.append("Alt")
        if modifiers & Qt.KeyboardModifier.ShiftModifier or key == Qt.Key.Key_Shift:
            parts.append("Shift")
        if modifiers & Qt.KeyboardModifier.MetaModifier or key == Qt.Key.Key_Meta:
            parts.append("Win")
            
        if not is_mod and key != Qt.Key.Key_unknown:
            key_name = QKeySequence(key).toString()
            if not key_name or len(key_name) > 15:
                special_keys = {
                    Qt.Key.Key_Escape: "Esc",
                    Qt.Key.Key_Tab: "Tab",
                    Qt.Key.Key_Backspace: "Backspace",
                    Qt.Key.Key_Return: "Enter",
                    Qt.Key.Key_Enter: "Enter",
                    Qt.Key.Key_Insert: "Insert",
                    Qt.Key.Key_Delete: "Delete",
                    Qt.Key.Key_Pause: "Pause",
                    Qt.Key.Key_Print: "Print Screen",
                    Qt.Key.Key_Home: "Home",
                    Qt.Key.Key_End: "End",
                    Qt.Key.Key_PageUp: "Page Up",
                    Qt.Key.Key_PageDown: "Page Down",
                    Qt.Key.Key_Up: "Up",
                    Qt.Key.Key_Down: "Down",
                    Qt.Key.Key_Left: "Left",
                    Qt.Key.Key_Right: "Right",
                    Qt.Key.Key_F1: "F1",
                    Qt.Key.Key_F2: "F2",
                    Qt.Key.Key_F3: "F3",
                    Qt.Key.Key_F4: "F4",
                    Qt.Key.Key_F5: "F5",
                    Qt.Key.Key_F6: "F6",
                    Qt.Key.Key_F7: "F7",
                    Qt.Key.Key_F8: "F8",
                    Qt.Key.Key_F9: "F9",
                    Qt.Key.Key_F10: "F10",
                    Qt.Key.Key_F11: "F11",
                    Qt.Key.Key_F12: "F12",
                    Qt.Key.Key_Space: "Space"
                }
                key_name = special_keys.get(key, f"Key_{key}")
            parts.append(key_name)
            
        if parts:
            self.recorded_text = "+".join(parts)
        else:
            self.recorded_text = ""
            
        self.update_display()
        event.accept()

    def validate_shortcut(self):
        if not self.recorded_text:
            return False, "Shortcut cannot be empty"
            
        parts = self.recorded_text.split("+")
        first_part = parts[0]
        if first_part not in ("Ctrl", "Alt", "Shift", "Win"):
            return False, "A shortcut should start with Windows key, Ctrl, Alt or Shift."
            
        last_part = parts[-1]
        if last_part in ("Ctrl", "Alt", "Shift", "Win"):
            return False, "Invalid shortcut"
            
        return True, ""

    def update_display(self):
        # Clear previous layout
        while self.box_layout.count():
            item = self.box_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        is_valid, msg = self.validate_shortcut()
        
        if is_valid:
            self.recording_box.setObjectName("recording_box_valid")
            self.recording_box.setStyleSheet(f"""
                QFrame#recording_box_valid {{
                    background-color: {DRACULA['background']};
                    border: 1px solid {DRACULA['comment']};
                    border-radius: 8px;
                }}
            """)
            self.warning_bar.hide()
            self.save_btn.setEnabled(True)
        else:
            self.recording_box.setObjectName("recording_box_invalid")
            self.recording_box.setStyleSheet(f"""
                QFrame#recording_box_invalid {{
                    background-color: rgba(255, 85, 85, 15);
                    border: 1.5px solid {DRACULA['red']};
                    border-radius: 8px;
                }}
            """)
            self.warning_msg.setText(msg)
            self.warning_bar.show()
            self.save_btn.setEnabled(False)
            
        if self.recorded_text:
            parts = self.recorded_text.split("+")
            self.box_layout.addStretch()
            for part in parts:
                lbl = QLabel(part)
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                if is_valid:
                    lbl_style = f"""
                        QLabel {{
                            background-color: {DRACULA['cyan']};
                            color: black;
                            font-weight: bold;
                            border-radius: 6px;
                            padding: 6px 14px;
                            font-family: 'Segoe UI', sans-serif;
                            font-size: 13px;
                        }}
                    """
                else:
                    lbl_style = f"""
                        QLabel {{
                            background-color: {DRACULA['red']};
                            color: white;
                            font-weight: bold;
                            border-radius: 6px;
                            padding: 6px 14px;
                            font-family: 'Segoe UI', sans-serif;
                            font-size: 13px;
                        }}
                    """
                lbl.setStyleSheet(lbl_style)
                self.box_layout.addWidget(lbl)
            self.box_layout.addStretch()
        else:
            self.box_layout.addStretch()
            lbl = QLabel("Press keys to record...")
            lbl.setStyleSheet(f"color: {DRACULA['comment']}; font-style: italic; font-size: 14px;")
            self.box_layout.addWidget(lbl)
            self.box_layout.addStretch()

    def reset_shortcut(self):
        self.recorded_text = self.current_shortcut
        self.update_display()

    def clear_shortcut(self):
        self.recorded_text = ""
        self.update_display()

def create_pencil_icon(color=QColor("white")) -> QIcon:
    pixmap = QPixmap(16, 16)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    
    pen = QPen(color, 1.5)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    
    # Draw a 45 degree pencil pointing down-right
    painter.drawLine(3, 11, 11, 3)
    painter.drawLine(5, 13, 13, 5)
    painter.drawLine(11, 3, 13, 5)
    painter.drawLine(3, 11, 1, 13)
    painter.drawLine(5, 13, 1, 13)
    
    painter.end()
    return QIcon(pixmap)

class ShortcutEditWidget(QWidget):
    def __init__(self, parent=None, current_shortcut=""):
        super().__init__(parent)
        self.current_shortcut = current_shortcut
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.init_ui()
        
    def init_ui(self):
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(8)
        
        self.keys_layout = QHBoxLayout()
        self.keys_layout.setContentsMargins(0, 0, 0, 0)
        self.keys_layout.setSpacing(4)
        self.main_layout.addLayout(self.keys_layout)
        
        self.edit_btn = QPushButton()
        self.edit_btn.setIcon(create_pencil_icon())
        self.edit_btn.setIconSize(QSize(12, 12))
        self.edit_btn.setFixedSize(24, 24)
        self.edit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.edit_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {DRACULA['current_line']};
                color: {DRACULA['foreground']};
                border: 1px solid {DRACULA['comment']};
                border-radius: 4px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: rgba(41, 182, 246, 45);
                border-color: {DRACULA['cyan']};
                color: {DRACULA['cyan']};
            }}
        """)
        self.edit_btn.clicked.connect(self.open_recorder)
        self.main_layout.addWidget(self.edit_btn)
        self.main_layout.addStretch()
        
        self.update_keys_display()
        
    def update_keys_display(self):
        while self.keys_layout.count():
            item = self.keys_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        if self.current_shortcut:
            parts = self.current_shortcut.split("+")
            for part in parts:
                lbl = QLabel(part)
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl.setCursor(Qt.CursorShape.PointingHandCursor)
                lbl.setStyleSheet(f"""
                    QLabel {{
                        background-color: {DRACULA['cyan']};
                        color: black;
                        font-weight: bold;
                        border-radius: 6px;
                        padding: 4px 10px;
                        font-family: 'Segoe UI', sans-serif;
                        font-size: 11px;
                        border: 1px solid {DRACULA['comment']};
                    }}
                """)
                self.keys_layout.addWidget(lbl)
        else:
            lbl = QLabel("None")
            lbl.setCursor(Qt.CursorShape.PointingHandCursor)
            lbl.setStyleSheet(f"color: {DRACULA['comment']}; font-style: italic;")
            self.keys_layout.addWidget(lbl)
            
    def text(self):
        return self.current_shortcut
        
    def setText(self, val):
        self.current_shortcut = val
        self.update_keys_display()
        
    def open_recorder(self):
        parent_dialog = self.window()
        if parent_dialog and isinstance(parent_dialog, QDialog):
            parent_dialog.setWindowOpacity(0.0)
            
        dlg = ShortcutRecorderDialog(self, self.current_shortcut)
        result = dlg.exec()
        
        if parent_dialog and isinstance(parent_dialog, QDialog):
            parent_dialog.setWindowOpacity(1.0)
            parent_dialog.raise_()
            parent_dialog.activateWindow()
            QApplication.processEvents()
            
        if result == QDialog.DialogCode.Accepted:
            self.setText(dlg.recorded_text)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.open_recorder()
            event.accept()
        else:
            super().mousePressEvent(event)

# ===============================
# DROP AREA FRAME FOR DRAG AND DROP
# ===============================

class DropAreaFrame(QFrame):
    files_dropped = pyqtSignal(list)
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("drop_area")
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Style
        self.setStyleSheet(f"""
            QFrame#drop_area {{
                border: 2px dashed rgba(98, 114, 164, 120);
                border-radius: 10px;
                background-color: rgba(40, 42, 54, 120);
                min-height: 90px;
            }}
            QFrame#drop_area:hover {{
                border-color: {DRACULA['cyan']};
                background-color: rgba(41, 182, 246, 30);
            }}
        """)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        
        self.label = QLabel("📥 Kéo thả file âm thanh vào đây hoặc Click để chọn file\n(Hỗ trợ: .wav, .mp3, .m4a, .flac, .ogg)")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet(f"""
            color: {DRACULA['foreground']};
            font-size: 12px;
            font-family: 'Segoe UI', sans-serif;
            background: transparent;
        """)
        self.layout.addWidget(self.label)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(f"""
                QFrame#drop_area {{
                    border: 2px dashed {DRACULA['cyan']};
                    background-color: rgba(41, 182, 246, 40);
                    min-height: 90px;
                }}
            """)

    def dragLeaveEvent(self, event):
        self.setStyleSheet(f"""
            QFrame#drop_area {{
                border: 2px dashed rgba(98, 114, 164, 120);
                border-radius: 10px;
                background-color: rgba(40, 42, 54, 120);
                min-height: 90px;
            }}
        """)

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_paths = [url.toLocalFile() for url in urls if url.isLocalFile()]
            if file_paths:
                self.files_dropped.emit(file_paths)
        self.setStyleSheet(f"""
            QFrame#drop_area {{
                border: 2px dashed rgba(98, 114, 164, 120);
                border-radius: 10px;
                background-color: rgba(40, 42, 54, 120);
                min-height: 90px;
            }}
        """)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

# ===============================
# IMAGE ZOOM DIALOG FOR OCR Snip
# ===============================

class ImageZoomDialog(QDialog):
    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Chi tiết ảnh chụp OCR")
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowCloseButtonHint)
        self.setStyleSheet(f"background-color: {DRACULA['background']}; color: {DRACULA['foreground']};")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Scroll area for large images
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        label = QLabel()
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        scroll.setWidget(label)
        layout.addWidget(scroll)
        
        # Adjust size based on pixmap, but limit max size
        w = min(pixmap.width() + 40, 1000)
        h = min(pixmap.height() + 40, 700)
        self.resize(w, h)

# ===============================
# SCREEN SELECTOR FOR OCR Snip
# ===============================

class ScreenSelector(QWidget):
    region_selected = pyqtSignal(int, int, int, int)
    closed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.SubWindow
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.start_pos = None
        self.end_pos = None
        self.is_dragging = False
        self.selection_made = False
        self.virtual_x_offset = 0
        self.virtual_y_offset = 0

    def start_selection(self):
        screen = QApplication.primaryScreen()
        if not screen:
            self.close()
            return
            
        virtual_geo = screen.virtualGeometry()
        self.setGeometry(virtual_geo)
        
        self.virtual_x_offset = virtual_geo.x()
        self.virtual_y_offset = virtual_geo.y()
        
        # Tạo ảnh ghép (pixmap) cho tất cả màn hình
        combined_pixmap = QPixmap(virtual_geo.width(), virtual_geo.height())
        combined_pixmap.fill(Qt.GlobalColor.black)
        
        painter = QPainter(combined_pixmap)
        screens = QApplication.screens()
        for s in screens:
            geom = s.geometry()
            screen_pixmap = s.grabWindow(0)
            painter.drawPixmap(geom.x() - virtual_geo.x(), geom.y() - virtual_geo.y(), screen_pixmap)
        painter.end()
        
        self.background_pixmap = combined_pixmap
        
        self.show()
        self.raise_()
        self.activateWindow()

    def paintEvent(self, event):
        painter = QPainter(self)
        
        painter.drawPixmap(0, 0, self.background_pixmap)
        
        painter.fillRect(self.rect(), QColor(0, 0, 0, 120))
        
        if self.start_pos and self.end_pos:
            rect = QRect(self.start_pos, self.end_pos).normalized()
            
            painter.drawPixmap(rect, self.background_pixmap, rect)
            
            pen = QPen(QColor(DRACULA["cyan"]), 2)
            painter.setPen(pen)
            painter.drawRect(rect)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_pos = event.position().toPoint()
            self.end_pos = self.start_pos
            self.is_dragging = True
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            self.end_pos = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.is_dragging:
            self.is_dragging = False
            rect = QRect(self.start_pos, self.end_pos).normalized()
            if rect.width() > 5 and rect.height() > 5:
                self.selection_made = True
                self.close()
                
                # Chuyển đổi tọa độ tương đối của widget thành tọa độ màn hình toàn cầu (global coordinate)
                global_x = rect.x() + self.virtual_x_offset
                global_y = rect.y() + self.virtual_y_offset
                
                # Tìm màn hình chứa vùng chọn để lấy hệ số scale (DPI) chính xác
                center_x = global_x + rect.width() // 2
                center_y = global_y + rect.height() // 2
                
                target_screen = QApplication.primaryScreen()
                for s in QApplication.screens():
                    if s.geometry().contains(center_x, center_y):
                        target_screen = s
                        break
                
                scale = target_screen.devicePixelRatio()
                
                self.region_selected.emit(
                    int(global_x),
                    int(global_y),
                    int(rect.width()),
                    int(rect.height())
                )
            else:
                self.selection_made = False
                self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.selection_made = False
            self.close()

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)

# ===============================
# FLOATING BUTTON WIDGET
# ===============================

class FloatingButton(QWidget):
    def __init__(self, parent_overlay):
        super().__init__()
        self.overlay = parent_overlay
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(54, 54)
        
        self.hovered = False
        
        # Load icon/logo
        self.logo_pixmap = QPixmap()
        icon_paths = [
            RESOURCE_DIR / "icon.png",
            BASE_DIR / "icon.png",
        ]
        for path in icon_paths:
            if path.exists():
                self.logo_pixmap.load(str(path))
                break
                
        self.drag_position = None
        
        # Place at the right edge and vertically centered of the screen by default
        screen = QApplication.primaryScreen().geometry()
        self.move(screen.width() - 60, (screen.height() - 54) // 2)

    def enterEvent(self, event):
        self.hovered = True
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.hovered = False
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Hover glow/shadow and layout styles
        if self.hovered:
            glow_color = QColor(41, 182, 246, 60)
            painter.setBrush(glow_color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(0, 0, 54, 54)
            
            bg_color = QColor(40, 42, 54, 240)
            border_color = QColor(41, 182, 246, 255)
            painter.setBrush(bg_color)
            painter.setPen(QPen(border_color, 1.5))
            painter.drawEllipse(4, 4, 46, 46)
        else:
            # Viền trong suốt (không vẽ đường viền) khi không hover
            bg_color = QColor(40, 42, 54, 180)
            painter.setBrush(bg_color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(4, 4, 46, 46)
        
        if not self.logo_pixmap.isNull():
            icon_size = 28 if self.hovered else 26
            icon_offset = (54 - icon_size) // 2
            target_rect = QRect(icon_offset, icon_offset, icon_size, icon_size)
            painter.drawPixmap(target_rect, self.logo_pixmap)
        else:
            painter.setPen(QColor("white"))
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Dịch")
            
        # Draw download progress circular ring if downloading
        if hasattr(self.overlay, "is_downloading") and self.overlay.is_downloading:
            progress = getattr(self.overlay, "download_progress", 0.0)
            pen_width = 3.0
            progress_color = QColor(41, 182, 246)
            pen = QPen(progress_color, pen_width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            
            rect = QRect(6, 6, 42, 42)
            start_angle = 90 * 16  # Top (12 o'clock)
            span_angle = -int((progress / 100.0) * 360.0 * 16.0)
            
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawArc(rect, start_angle, span_angle)
            
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self.drag_start_pos = event.globalPosition().toPoint()
            self.drag_start_time = time.time()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self.drag_position is not None:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = None
            if hasattr(self, "drag_start_time") and hasattr(self, "drag_start_pos"):
                click_duration = time.time() - self.drag_start_time
                displacement = (event.globalPosition().toPoint() - self.drag_start_pos).manhattanLength()
                if click_duration < 0.25 and displacement < 6:
                    self.on_clicked()
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            self.show_context_menu(event.globalPosition().toPoint())
            event.accept()

    def show_context_menu(self, pos):
        menu = QMenu(self)
        
        # Style the menu to match the dark theme
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {DRACULA['background']};
                color: {DRACULA['foreground']};
                border: 1px solid {DRACULA['comment']};
                border-radius: 6px;
                padding: 4px 0px;
            }}
            QMenu::item {{
                padding: 6px 20px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 12px;
            }}
            QMenu::item:selected {{
                background-color: {DRACULA['current_line']};
            }}
        """)
        
        show_action = QAction("Hiện ứng dụng", self)
        show_action.triggered.connect(self.on_clicked)
        menu.addAction(show_action)
        
        settings_action = QAction("Mở Cài đặt", self)
        settings_action.triggered.connect(self.open_settings)
        menu.addAction(settings_action)
        
        menu.addSeparator()
        
        quit_action = QAction("Thoát", self)
        quit_action.triggered.connect(self.overlay.quit_app)
        menu.addAction(quit_action)
        
        menu.exec(pos)

    def open_settings(self):
        self.hide()
        self.overlay.show_app()
        self.overlay.open_settings_dialog()

    def on_clicked(self):
        self.hide()
        self.overlay.show_app()

# ===============================
# IMAGE DROP AREA FOR PROMPT GEN
# ===============================

class ImageDropArea(QFrame):
    image_dropped = pyqtSignal(QPixmap)
    clicked = pyqtSignal()
    delete_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("image_drop_area")
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(220, 220)
        self.current_pixmap = None
        
        self.setStyleSheet(f"""
            QFrame#image_drop_area {{
                border: 2px dashed rgba(98, 114, 164, 120);
                border-radius: 10px;
                background-color: rgba(40, 42, 54, 120);
            }}
            QFrame#image_drop_area:hover {{
                border-color: {DRACULA['cyan']};
                background-color: rgba(41, 182, 246, 30);
            }}
        """)
        
        from PyQt6.QtWidgets import QGridLayout, QHBoxLayout, QPushButton, QWidget
        self.grid_layout = QGridLayout(self)
        self.grid_layout.setContentsMargins(5, 5, 5, 5)
        
        self.label = QLabel("📥 Kéo thả ảnh,\nClick chọn hoặc\nPaste ảnh vào đây")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setStyleSheet(f"""
            color: {DRACULA['foreground']};
            font-size: 11px;
            font-family: 'Segoe UI', sans-serif;
            background: transparent;
        """)
        self.grid_layout.addWidget(self.label, 0, 0)
        
        # Create a container widget for overlays to guarantee correct alignment
        self.overlay_widget = QWidget(self)
        self.overlay_widget.setObjectName("overlay_widget")
        self.overlay_widget.setStyleSheet("background: transparent; border: none;")
        
        self.overlay_layout = QHBoxLayout(self.overlay_widget)
        self.overlay_layout.setContentsMargins(0, 0, 0, 0)
        self.overlay_layout.setSpacing(4)
        
        # Paste Button - Always visible! Placed on the top-left corner
        self.paste_btn = QPushButton("📋", self)
        self.paste_btn.setToolTip("Dán ảnh từ Clipboard (Ctrl+V)")
        self.paste_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.paste_btn.setFixedSize(24, 24)
        self.paste_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(0, 0, 0, 160);
                color: white;
                border: 1px solid {DRACULA['comment']};
                border-radius: 12px;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: rgba(241, 250, 140, 180);
                border-color: {DRACULA['yellow']};
            }}
            QPushButton:disabled {{
                background-color: rgba(68, 71, 90, 50);
                color: rgba(248, 248, 242, 60);
                border: 1px solid rgba(98, 114, 164, 30);
            }}
        """)
        
        # Overlay Copy Button - Changed icon to 📄
        self.copy_btn = QPushButton("📄")
        self.copy_btn.setToolTip("Copy ảnh vào Clipboard")
        self.copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.copy_btn.setFixedSize(24, 24)
        self.copy_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(0, 0, 0, 160);
                color: white;
                border: 1px solid {DRACULA['comment']};
                border-radius: 12px;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: rgba(189, 147, 249, 200);
                border-color: {DRACULA['purple']};
            }}
        """)
        self.copy_btn.hide()
        self.copy_btn.clicked.connect(self.copy_image_to_clipboard)
        self.overlay_layout.addWidget(self.copy_btn)
        
        # Overlay Save Button
        self.save_btn = QPushButton("💾")
        self.save_btn.setToolTip("Tải ảnh về máy")
        self.save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_btn.setFixedSize(24, 24)
        self.save_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(0, 0, 0, 160);
                color: white;
                border: 1px solid {DRACULA['comment']};
                border-radius: 12px;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: rgba(80, 250, 123, 200);
                border-color: {DRACULA['green']};
            }}
        """)
        self.save_btn.hide()
        self.save_btn.clicked.connect(self.save_image_to_file)
        self.overlay_layout.addWidget(self.save_btn)
        
        # Overlay Delete Button
        self.delete_btn = QPushButton("🗑️")
        self.delete_btn.setToolTip("Xóa ảnh tham chiếu")
        self.delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete_btn.setFixedSize(24, 24)
        self.delete_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(0, 0, 0, 160);
                color: white;
                border: 1px solid {DRACULA['comment']};
                border-radius: 12px;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: rgba(255, 85, 85, 200);
                border-color: {DRACULA['red']};
            }}
        """)
        self.delete_btn.hide()
        self.delete_btn.clicked.connect(self.delete_clicked.emit)
        self.overlay_layout.addWidget(self.delete_btn)
        
        # Copy system prompt button in bottom-right corner of the image area
        self.copy_sys_prompt_btn = QPushButton("📋", self)
        self.copy_sys_prompt_btn.setToolTip("Copy System Prompt chuẩn bị gửi đi")
        self.copy_sys_prompt_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.copy_sys_prompt_btn.setFixedSize(24, 24)
        self.copy_sys_prompt_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(0, 0, 0, 160);
                color: white;
                border: 1px solid {DRACULA['comment']};
                border-radius: 12px;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: rgba(41, 182, 246, 200);
                border-color: {DRACULA['cyan']};
            }}
        """)
        self.copy_sys_prompt_btn.hide()
        
        # Position the overlay container widget absolutely in the top-right corner
        self.overlay_widget.adjustSize()
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.paste_btn.move(8, 8)
        self.paste_btn.raise_()
        
        self.overlay_widget.adjustSize()
        w = self.overlay_widget.width()
        self.overlay_widget.move(self.width() - w - 8, 8)
        self.overlay_widget.raise_()
        
        self.copy_sys_prompt_btn.move(self.width() - self.copy_sys_prompt_btn.width() - 8, self.height() - self.copy_sys_prompt_btn.height() - 8)
        self.copy_sys_prompt_btn.raise_()
        
    def showEvent(self, event):
        super().showEvent(event)
        self.paste_btn.move(8, 8)
        self.paste_btn.raise_()
        
        self.overlay_widget.adjustSize()
        w = self.overlay_widget.width()
        self.overlay_widget.move(self.width() - w - 8, 8)
        self.overlay_widget.raise_()
        
        self.copy_sys_prompt_btn.move(self.width() - self.copy_sys_prompt_btn.width() - 8, self.height() - self.copy_sys_prompt_btn.height() - 8)
        self.copy_sys_prompt_btn.raise_()
        
    def set_pixmap(self, pixmap):
        self.current_pixmap = pixmap
        if pixmap and not pixmap.isNull():
            self.copy_btn.show()
            self.save_btn.show()
            self.delete_btn.show()
            self.copy_sys_prompt_btn.show()
        else:
            self.copy_btn.hide()
            self.save_btn.hide()
            self.delete_btn.hide()
            self.copy_sys_prompt_btn.hide()
            
        self.paste_btn.move(8, 8)
        self.paste_btn.raise_()
        
        self.overlay_widget.adjustSize()
        w = self.overlay_widget.width()
        self.overlay_widget.move(self.width() - w - 8, 8)
        self.overlay_widget.raise_()
        
        self.copy_sys_prompt_btn.move(self.width() - self.copy_sys_prompt_btn.width() - 8, self.height() - self.copy_sys_prompt_btn.height() - 8)
        self.copy_sys_prompt_btn.raise_()
            
    def copy_image_to_clipboard(self):
        if self.current_pixmap and not self.current_pixmap.isNull():
            from PyQt6.QtWidgets import QApplication
            from PyQt6.QtCore import QTimer
            clipboard = QApplication.clipboard()
            clipboard.setImage(self.current_pixmap.toImage())
            self.copy_btn.setText("✅")
            self.copy_btn.setToolTip("Đã copy ảnh!")
            QTimer.singleShot(1500, lambda: (self.copy_btn.setText("📄"), self.copy_btn.setToolTip("Copy ảnh vào Clipboard")))

    def save_image_to_file(self):
        if self.current_pixmap and not self.current_pixmap.isNull():
            from PyQt6.QtWidgets import QFileDialog, QMessageBox
            from PyQt6.QtCore import QTimer
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Lưu ảnh tham chiếu",
                "",
                "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
            )
            if file_path:
                try:
                    self.current_pixmap.save(file_path)
                    self.save_btn.setText("✅")
                    self.save_btn.setToolTip("Đã lưu ảnh!")
                    QTimer.singleShot(1500, lambda: (self.save_btn.setText("💾"), self.save_btn.setToolTip("Tải ảnh về máy")))
                except Exception as e:
                    QMessageBox.warning(self, "Lỗi", f"Không thể lưu ảnh: {str(e)}")
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() or event.mimeData().hasImage():
            event.acceptProposedAction()
            self.setStyleSheet(f"""
                QFrame#image_drop_area {{
                    border: 2px dashed {DRACULA['cyan']};
                    background-color: rgba(41, 182, 246, 40);
                }}
            """)
            
    def dragLeaveEvent(self, event):
        self.setStyleSheet(f"""
            QFrame#image_drop_area {{
                border: 2px dashed rgba(98, 114, 164, 120);
                border-radius: 10px;
                background-color: rgba(40, 42, 54, 120);
            }}
        """)
        
    def dropEvent(self, event):
        mime_data = event.mimeData()
        pixmap = None
        if mime_data.hasUrls():
            urls = mime_data.urls()
            if urls:
                file_path = urls[0].toLocalFile()
                if file_path and os.path.exists(file_path):
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"]:
                        pixmap = QPixmap(file_path)
        elif mime_data.hasImage():
            image = mime_data.imageData()
            pixmap = QPixmap.fromImage(image)
            
        if pixmap and not pixmap.isNull():
            self.image_dropped.emit(pixmap)
            
        self.setStyleSheet(f"""
            QFrame#image_drop_area {{
                border: 2px dashed rgba(98, 114, 164, 120);
                border-radius: 10px;
                background-color: rgba(40, 42, 54, 120);
            }}
        """)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if (self.copy_btn.underMouse() or 
                self.save_btn.underMouse() or 
                self.delete_btn.underMouse() or 
                self.paste_btn.underMouse()):
                return
            self.clicked.emit()

# ===============================
# PROMPT GENERATOR WIDGET (TAB 5)
# ===============================

class PromptGenWidget(QWidget):
    prompt_success_signal = pyqtSignal(str)
    prompt_error_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_overlay = parent
        self.image_pixmap = None
        self.analyzing = False
        self.temp_image_path = None
        
        self.prompt_success_signal.connect(self.on_success)
        self.prompt_error_signal.connect(self.on_error)
        
        self.init_ui()
        
    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)
        
        # Left Panel (Image preview & controls)
        left_panel = QVBoxLayout()
        left_panel.setSpacing(6)
        left_panel.setContentsMargins(0, 0, 0, 0)
        
        self.image_area = ImageDropArea(self)
        self.image_area.image_dropped.connect(self.set_image)
        self.image_area.clicked.connect(self.browse_image)
        self.image_area.delete_clicked.connect(self.clear_image)
        left_panel.addWidget(self.image_area, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        
        # Load settings
        settings = load_settings()
        
        # Checkbox: Reference original image
        self.ref_image_checkbox = QCheckBox("Ref ảnh từ ảnh gốc")
        self.ref_image_checkbox.setFixedWidth(220)
        self.ref_image_checkbox.setChecked(settings.get("prompt_ref_image", True))
        self.ref_image_checkbox.setStyleSheet(f"""
            QCheckBox {{
                color: {DRACULA['foreground']};
                font-size: 11px;
                font-family: 'Segoe UI', sans-serif;
            }}
            QCheckBox::indicator {{
                width: 14px;
                height: 14px;
            }}
        """)
        left_panel.addWidget(self.ref_image_checkbox)

        # Checkbox: Reference background from reference photo
        self.ref_bg_checkbox = QCheckBox("Ref background từ reference photo")
        self.ref_bg_checkbox.setFixedWidth(220)
        self.ref_bg_checkbox.setChecked(settings.get("prompt_ref_bg", False))
        self.ref_bg_checkbox.setStyleSheet(f"""
            QCheckBox {{
                color: {DRACULA['foreground']};
                font-size: 11px;
                font-family: 'Segoe UI', sans-serif;
            }}
            QCheckBox::indicator {{
                width: 14px;
                height: 14px;
            }}
        """)
        left_panel.addWidget(self.ref_bg_checkbox)
        
        # Checkbox: Keep Hair Style
        self.keep_hair_checkbox = QCheckBox("Giữ kiểu tóc gốc")
        self.keep_hair_checkbox.setFixedWidth(220)
        self.keep_hair_checkbox.setChecked(settings.get("prompt_keep_hair", True))
        self.keep_hair_checkbox.setStyleSheet(f"""
            QCheckBox {{
                color: {DRACULA['foreground']};
                font-size: 11px;
                font-family: 'Segoe UI', sans-serif;
            }}
            QCheckBox::indicator {{
                width: 14px;
                height: 14px;
            }}
        """)
        left_panel.addWidget(self.keep_hair_checkbox)
        
        # Checkbox: Keep Accessories Style (Loại bỏ phụ kiện / Giữ phụ kiện)
        self.keep_acc_checkbox = QCheckBox("Loại bỏ phụ kiện trên người")
        self.keep_acc_checkbox.setFixedWidth(220)
        self.keep_acc_checkbox.setChecked(settings.get("prompt_remove_acc", False))
        self.keep_acc_checkbox.setStyleSheet(f"""
            QCheckBox {{
                color: {DRACULA['foreground']};
                font-size: 11px;
                font-family: 'Segoe UI', sans-serif;
            }}
            QCheckBox::indicator {{
                width: 14px;
                height: 14px;
            }}
        """)
        left_panel.addWidget(self.keep_acc_checkbox)
        
        # ComboBox: Model profile Selection
        self.model_profile_select = QComboBox()
        self.model_profile_select.setFixedWidth(220)
        self.model_profile_select.setCursor(Qt.CursorShape.PointingHandCursor)
        self.model_profile_select.setStyleSheet(f"""
            QComboBox {{
                color: {DRACULA['foreground']};
                background-color: {DRACULA['current_line']};
                border: 1px solid {DRACULA['comment']};
                border-radius: 6px;
                padding: 4px 20px 4px 10px;
                font-size: 11px;
                font-family: 'Segoe UI', sans-serif;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox QAbstractItemView {{
                color: {DRACULA['foreground']};
                background-color: {DRACULA['background']};
                selection-background-color: {DRACULA['current_line']};
                border: 1px solid {DRACULA['comment']};
                border-radius: 6px;
            }}
        """)
        left_panel.addWidget(self.model_profile_select)
        self.refresh_model_profiles()
        
        # ComboBox: Template Mode Selection
        self.template_select = QComboBox()
        self.template_select.setFixedWidth(220)
        self.template_select.addItems(["Mặc định", "JSON Chi Tiết"])
        saved_template = settings.get("prompt_template_mode", "Mặc định")
        idx_t = self.template_select.findText(saved_template)
        if idx_t >= 0:
            self.template_select.setCurrentIndex(idx_t)
        self.template_select.setCursor(Qt.CursorShape.PointingHandCursor)
        self.template_select.setStyleSheet(f"""
            QComboBox {{
                color: {DRACULA['foreground']};
                background-color: {DRACULA['current_line']};
                border: 1px solid {DRACULA['comment']};
                border-radius: 6px;
                padding: 4px 20px 4px 10px;
                font-size: 11px;
                font-family: 'Segoe UI', sans-serif;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox QAbstractItemView {{
                color: {DRACULA['foreground']};
                background-color: {DRACULA['background']};
                selection-background-color: {DRACULA['current_line']};
                border: 1px solid {DRACULA['comment']};
                border-radius: 6px;
            }}
        """)
        left_panel.addWidget(self.template_select)
        
        # Input: Custom Prompt Suggestions container to overlay "×" clear button
        self.input_container = QFrame()
        self.input_container.setFixedWidth(220)
        self.input_container.setFixedHeight(100)
        self.input_container.setStyleSheet("background: transparent; border: none;")
        
        from PyQt6.QtWidgets import QGridLayout
        input_grid = QGridLayout(self.input_container)
        input_grid.setContentsMargins(0, 0, 0, 0)
        
        self.custom_prompt_input = QTextEdit()
        self.custom_prompt_input.setPlaceholderText("Gợi ý: background ở đâu, pose thế nào, trang phục gì...")
        self.custom_prompt_input.setFixedHeight(100)
        self.custom_prompt_input.setStyleSheet(f"""
            QTextEdit {{
                color: {DRACULA['foreground']};
                background-color: rgba(40, 42, 54, 120);
                border: 1px solid rgba(98, 114, 164, 60);
                border-radius: 6px;
                padding: 4px;
                padding-right: 24px;
                font-size: 11px;
                font-family: 'Segoe UI', sans-serif;
            }}
        """)
        input_grid.addWidget(self.custom_prompt_input, 0, 0)
        
        # Small "×" clear button overlay
        self.clear_input_btn = QPushButton("×")
        self.clear_input_btn.setToolTip("Xóa gợi ý")
        self.clear_input_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_input_btn.setFixedSize(18, 18)
        self.clear_input_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {DRACULA['comment']};
                border: none;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                color: {DRACULA['red']};
            }}
        """)
        self.clear_input_btn.hide()
        self.clear_input_btn.clicked.connect(self.custom_prompt_input.clear)
        input_grid.addWidget(self.clear_input_btn, 0, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        
        # Small "📋" copy system prompt button overlay at bottom-right corner
        self.copy_sys_prompt_btn = QPushButton("📋")
        self.copy_sys_prompt_btn.setToolTip("Copy System Prompt chuẩn bị gửi đi")
        self.copy_sys_prompt_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.copy_sys_prompt_btn.setFixedSize(18, 18)
        self.copy_sys_prompt_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {DRACULA['comment']};
                border: none;
                font-size: 11px;
                font-weight: bold;
                margin-right: 3px;
                margin-bottom: 3px;
            }}
            QPushButton:hover {{
                color: {DRACULA['cyan']};
            }}
        """)
        self.copy_sys_prompt_btn.clicked.connect(self.copy_system_prompt)
        input_grid.addWidget(self.copy_sys_prompt_btn, 0, 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)
        # Connect textChanged signal to show/hide the clear button
        self.custom_prompt_input.textChanged.connect(
            lambda: self.clear_input_btn.setVisible(bool(self.custom_prompt_input.toPlainText().strip()))
        )
        
        left_panel.addWidget(self.input_container)
        
        # Connect change signals to saving settings
        self.ref_image_checkbox.stateChanged.connect(self.save_prompt_settings)
        self.ref_bg_checkbox.stateChanged.connect(self.save_prompt_settings)
        self.keep_hair_checkbox.stateChanged.connect(self.save_prompt_settings)
        self.keep_acc_checkbox.stateChanged.connect(self.save_prompt_settings)
        self.model_profile_select.currentIndexChanged.connect(self.save_prompt_settings)
        self.template_select.currentIndexChanged.connect(self.save_prompt_settings)
        self.custom_prompt_input.textChanged.connect(self.update_analyze_btn_state)

        # Button: Paste Image (hidden, replaced by overlay button in drop area)
        self.paste_btn = QPushButton()
        self.paste_btn.hide()
        self.paste_btn.clicked.connect(self.paste_image)
        self.image_area.paste_btn.clicked.connect(self.paste_image)
        self.image_area.copy_sys_prompt_btn.clicked.connect(self.copy_system_prompt)
        
        # Button: Delete Image (hidden, replaced by overlay button in drop area)
        self.delete_btn = QPushButton()
        self.delete_btn.hide()
        self.delete_btn.clicked.connect(self.clear_image)
        
        # Button: Analyze Image
        self.analyze_btn = QPushButton("✨ Phân tích ảnh")
        self.analyze_btn.setFixedWidth(220)
        self.analyze_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet(f"""
            QPushButton {{
                color: white;
                background-color: {DRACULA['btn_green']};
                border: 1px solid {DRACULA['btn_green']};
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 12px;
                font-weight: bold;
                font-family: 'Segoe UI', sans-serif;
            }}
            QPushButton:hover {{
                background-color: {DRACULA['btn_green_hover']};
            }}
            QPushButton:disabled {{
                background-color: rgba(68, 71, 90, 50);
                color: rgba(248, 248, 242, 60);
                border: 1px solid rgba(98, 114, 164, 30);
            }}
        """)
        self.analyze_btn.clicked.connect(self.start_analysis)
        left_panel.addWidget(self.analyze_btn)
        
        # Button: Screen Capture
        self.capture_btn = QPushButton("📷 Quét màn hình")
        self.capture_btn.setFixedWidth(220)
        self.capture_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.capture_btn.setStyleSheet(f"""
            QPushButton {{
                color: white;
                background-color: rgba(41, 182, 246, 150);
                border: 1px solid {DRACULA['cyan']};
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 12px;
                font-weight: bold;
                font-family: 'Segoe UI', sans-serif;
            }}
            QPushButton:hover {{
                background-color: rgba(41, 182, 246, 200);
            }}
        """)
        self.capture_btn.clicked.connect(self.start_screen_capture)
        left_panel.addWidget(self.capture_btn)
        
        left_panel.addStretch()
        main_layout.addLayout(left_panel, 0)
        
        # Right Panel (Output prompt result & Copy button)
        right_panel = QVBoxLayout()
        right_panel.setSpacing(6)
        right_panel.setContentsMargins(0, 0, 0, 0)
        
        # Top row of right panel: Copy Button
        right_top_layout = QHBoxLayout()
        right_top_layout.setContentsMargins(4, 2, 4, 2)
        right_top_layout.addStretch()
        
        self.copy_prompt_btn = QPushButton("📋")
        self.copy_prompt_btn.setFixedSize(28, 28)
        self.copy_prompt_btn.setToolTip("Copy Cả Hai")
        self.copy_prompt_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.copy_prompt_btn.setEnabled(False)
        self.copy_prompt_btn.setStyleSheet(f"""
            QPushButton {{
                color: {DRACULA['foreground']};
                background-color: {DRACULA['current_line']};
                border: 1px solid {DRACULA['comment']};
                border-radius: 14px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {DRACULA['cyan']};
                color: {DRACULA['background']};
                border-color: {DRACULA['cyan']};
            }}
            QPushButton:disabled {{
                background-color: rgba(68, 71, 90, 50);
                color: rgba(248, 248, 242, 60);
                border: 1px solid rgba(98, 114, 164, 30);
            }}
        """)
        self.copy_prompt_btn.clicked.connect(self.copy_prompt)
        right_top_layout.addWidget(self.copy_prompt_btn)
        
        self.save_json_btn = QPushButton("💾")
        self.save_json_btn.setFixedSize(28, 28)
        self.save_json_btn.setToolTip("Lưu JSON")
        self.save_json_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_json_btn.setEnabled(False)
        self.save_json_btn.setStyleSheet(f"""
            QPushButton {{
                color: {DRACULA['foreground']};
                background-color: {DRACULA['current_line']};
                border: 1px solid {DRACULA['comment']};
                border-radius: 14px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {DRACULA['green']};
                color: {DRACULA['background']};
                border-color: {DRACULA['green']};
            }}
            QPushButton:disabled {{
                background-color: rgba(68, 71, 90, 50);
                color: rgba(248, 248, 242, 60);
                border: 1px solid rgba(98, 114, 164, 30);
            }}
        """)
        self.save_json_btn.clicked.connect(self.save_json_file)
        right_top_layout.addWidget(self.save_json_btn)
        
        self.delete_result_btn = QPushButton("🗑️")
        self.delete_result_btn.setFixedSize(28, 28)
        self.delete_result_btn.setToolTip("Xóa kết quả")
        self.delete_result_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete_result_btn.setEnabled(False)
        self.delete_result_btn.setStyleSheet(f"""
            QPushButton {{
                color: white;
                background-color: rgba(255, 85, 85, 120);
                border: 1px solid {DRACULA['red']};
                border-radius: 14px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: rgba(255, 85, 85, 200);
                border-color: {DRACULA['red']};
            }}
            QPushButton:disabled {{
                background-color: rgba(68, 71, 90, 50);
                color: rgba(248, 248, 242, 60);
                border: 1px solid rgba(98, 114, 164, 30);
            }}
        """)
        self.delete_result_btn.clicked.connect(self.clear_result)
        right_top_layout.addWidget(self.delete_result_btn)
        
        # Right Panel: QTabWidget holding 'Prompt Engineering' and 'Prompt'
        self.right_tab_widget = QTabWidget()
        self.right_tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid rgba(98, 114, 164, 50);
                background: transparent;
                border-radius: 8px;
            }}
            QTabBar::tab {{
                background: rgba(40, 42, 54, 150);
                color: {DRACULA['comment']};
                border: 1px solid rgba(98, 114, 164, 50);
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: bold;
                font-family: 'Segoe UI', sans-serif;
                margin-right: 4px;
            }}
            QTabBar::tab:hover {{
                background: {DRACULA['current_line']};
                color: {DRACULA['foreground']};
            }}
            QTabBar::tab:selected {{
                background: rgba(40, 42, 54, 100);
                color: {DRACULA['cyan']};
                border-color: rgba(98, 114, 164, 50);
            }}
        """)
        
        # Tab 1 of Right Panel: Prompt Engineering (Detailed JSON result)
        pe_widget = QWidget()
        pe_layout = QVBoxLayout(pe_widget)
        pe_layout.setContentsMargins(4, 4, 4, 4)
        pe_layout.setSpacing(4)
        
        self.result_view = QTextBrowser()
        self.result_view.setReadOnly(True)
        self.result_view.setOpenLinks(False)
        self.result_view.anchorClicked.connect(self.handle_link_clicked)
        self.result_view.setStyleSheet(f"""
            QTextBrowser {{
                color: {DRACULA['foreground']};
                background-color: rgba(40, 42, 54, 100);
                border: none;
                padding: 6px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 12px;
            }}
            QScrollBar:vertical {{
                border: none;
                background: rgba(40, 42, 54, 30);
                width: 6px;
                border-radius: 3px;
            }}
            QScrollBar::handle:vertical {{
                background: rgba(98, 114, 164, 150);
                border-radius: 3px;
            }}
        """)
        self.result_view.setPlaceholderText("Kéo thả/Dán ảnh và nhấp vào 'Phân tích ảnh' để tạo prompt...")
        pe_layout.addLayout(right_top_layout)
        pe_layout.addWidget(self.result_view)
        self.right_tab_widget.addTab(pe_widget, "Prompt Engineering")
        
        # Tab 2 of Right Panel: Prompt (List of parsed values from JSON)
        self.prompt_list_tab = QWidget()
        prompt_list_layout = QVBoxLayout(self.prompt_list_tab)
        prompt_list_layout.setContentsMargins(4, 4, 4, 4)
        prompt_list_layout.setSpacing(4)
        
        self.prompt_list_view = QTextBrowser()
        self.prompt_list_view.setReadOnly(True)
        self.prompt_list_view.setOpenLinks(False)
        self.prompt_list_view.anchorClicked.connect(self.handle_link_clicked)
        self.prompt_list_view.setStyleSheet(f"""
            QTextBrowser {{
                color: {DRACULA['foreground']};
                background-color: rgba(40, 42, 54, 100);
                border: none;
                padding: 6px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 12px;
            }}
            QScrollBar:vertical {{
                border: none;
                background: rgba(40, 42, 54, 30);
                width: 6px;
                border-radius: 3px;
            }}
            QScrollBar::handle:vertical {{
                background: rgba(98, 114, 164, 150);
                border-radius: 3px;
            }}
        """)
        self.prompt_list_view.setPlaceholderText("Danh sách các trường phân tích chi tiết (Subject, Hair, Clothing...) sẽ hiển thị ở đây...")
        prompt_list_layout.addWidget(self.prompt_list_view)
        self.right_tab_widget.addTab(self.prompt_list_tab, "Prompt")
        
        right_panel.addWidget(self.right_tab_widget)
        
        main_layout.addLayout(right_panel, 1)

        # Keyboard shortcuts (Only active when this tab/widget or its children has focus)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        self.shortcut_space = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self.shortcut_space.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.shortcut_space.activated.connect(self.start_analysis)
        
        self.shortcut_paste = QShortcut(QKeySequence("Ctrl+V"), self)
        self.shortcut_paste.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.shortcut_paste.activated.connect(self.paste_image)
        
        self.shortcut_delete = QShortcut(QKeySequence(Qt.Key.Key_Delete), self)
        self.shortcut_delete.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.shortcut_delete.activated.connect(self.clear_image)

    def keyPressEvent(self, event):
        super().keyPressEvent(event)

    def paste_image(self):
        if self.analyzing:
            return
            
        clipboard = QApplication.clipboard()
        mime_data = clipboard.mimeData()
        pixmap = None
        
        if mime_data.hasImage():
            image = clipboard.image()
            pixmap = QPixmap.fromImage(image)
        elif mime_data.hasUrls():
            urls = mime_data.urls()
            if urls:
                file_path = urls[0].toLocalFile()
                if file_path and os.path.exists(file_path):
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"]:
                        pixmap = QPixmap(file_path)
                        
        if pixmap is None or pixmap.isNull():
            QMessageBox.information(
                self, 
                "Thông báo", 
                "Không tìm thấy ảnh hoặc tệp ảnh hợp lệ trong Clipboard. Vui lòng copy ảnh hoặc file ảnh trước."
            )
            return
            
        self.set_image(pixmap)
        
    def set_image(self, pixmap):
        self.image_pixmap = pixmap
        scaled = pixmap.scaled(
            210, 210,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_area.label.setPixmap(scaled)
        self.image_area.set_pixmap(pixmap)
        self.update_analyze_btn_state()
        self.delete_btn.setEnabled(True)
        
    def clear_image(self):
        self.image_pixmap = None
        self.image_area.label.clear()
        self.image_area.label.setText("📥 Kéo thả ảnh,\nClick chọn hoặc\nPaste ảnh vào đây")
        self.image_area.set_pixmap(None)
        self.update_analyze_btn_state()
        self.delete_btn.setEnabled(False)
        self.custom_prompt_input.clear()
        self.clear_result()
        
    def clear_result(self):
        self.result_view.clear()
        self.prompt_en = ""
        self.prompt_vi = ""
        self.delete_result_btn.setEnabled(False)
        self.copy_prompt_btn.setEnabled(False)
        self.save_json_btn.setEnabled(False)
        
    def copy_system_prompt(self):
        use_ref = self.ref_image_checkbox.isChecked() and self.image_pixmap is not None
        keep_hair = self.keep_hair_checkbox.isChecked()
        keep_acc = not self.keep_acc_checkbox.isChecked()
        template_mode = "json" if self.template_select.currentText() == "JSON Chi Tiết" else "default"
        selected_model = self.model_profile_select.currentText()
        user_suggestions = self.custom_prompt_input.toPlainText().strip()
        
        prompt = build_gemini_system_prompt(use_ref, template_mode, keep_hair, keep_acc, selected_model, user_suggestions)
        
        clipboard = QApplication.clipboard()
        clipboard.setText(prompt)
        QToolTip.showText(QCursor.pos(), "Đã copy System Prompt chuẩn bị gửi đi!")
        
    def browse_image(self):
        if self.analyzing:
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn ảnh tham chiếu",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.webp *.gif);;All Files (*)"
        )
        if file_path:
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                self.set_image(pixmap)
                
    def start_screen_capture(self):
        if self.analyzing:
            return
            
        if not USE_GEMINI or not gemini_chat:
            QMessageBox.warning(
                self,
                "Cảnh báo",
                "Chưa kết nối Gemini Web. Vui lòng kết nối Profile ở nút hình người 👤."
            )
            return
        
        self.parent_overlay.hide_app()
        self.selector = ScreenSelector()
        
        def on_region_selected(x, y, w, h):
            self.parent_overlay.show_app()
            
            screen = QGuiApplication.screenAt(QPoint(x, y))
            if screen is None:
                screen = QGuiApplication.primaryScreen()
            if not screen:
                return
                
            local_x = x - screen.geometry().x()
            local_y = y - screen.geometry().y()
            pixmap = screen.grabWindow(0, local_x, local_y, w, h)
            if pixmap and not pixmap.isNull():
                self.set_image(pixmap)
                # Immediately analyze!
                self.start_analysis()
                
        def on_selector_closed():
            self.parent_overlay.show_app()
            
        self.selector.region_selected.connect(on_region_selected)
        self.selector.closed.connect(on_selector_closed)
        self.selector.start_selection()

    def start_analysis(self):
        user_suggestions = self.custom_prompt_input.toPlainText().strip()
        if (not self.image_pixmap and not user_suggestions) or self.analyzing:
            return
            
        if not USE_GEMINI or not gemini_chat:
            QMessageBox.warning(
                self,
                "Cảnh báo",
                "Chưa kết nối Gemini Web. Vui lòng kết nối Profile ở nút hình người 👤."
            )
            return
            
        self.analyzing = True
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("⏳ Đang phân tích...")
        self.capture_btn.setEnabled(False)
        self.delete_btn.setEnabled(False)
        self.delete_result_btn.setEnabled(False)
        self.paste_btn.setEnabled(False)
        self.copy_prompt_btn.setEnabled(False)
        self.save_json_btn.setEnabled(False)
        self.custom_prompt_input.setEnabled(False)
        
        self.result_view.setHtml(f"<span style='color: {DRACULA['cyan']}; font-weight: bold;'>⏳ Đang phân tích ảnh bằng Gemini... Vui lòng đợi.</span>")
        
        temp_image_path = None
        if self.image_pixmap:
            # Save image to temp file
            temp_dir = tempfile.gettempdir()
            self.temp_image_path = os.path.join(temp_dir, f"prompt_ref_{os.getpid()}.png")
            self.image_pixmap.save(self.temp_image_path, "PNG")
            temp_image_path = self.temp_image_path
        else:
            self.temp_image_path = None
        
        use_ref = self.ref_image_checkbox.isChecked() and self.image_pixmap is not None
        keep_hair = self.keep_hair_checkbox.isChecked()
        keep_acc = not self.keep_acc_checkbox.isChecked() # keep_acc is True if checkbox is unchecked (meaning we do not remove accessories)
        keep_bg = self.ref_bg_checkbox.isChecked()
        template_mode = "json" if self.template_select.currentText() == "JSON Chi Tiết" else "default"
        selected_model = self.model_profile_select.currentText()
        user_suggestions = self.custom_prompt_input.toPlainText().strip()
        
        def run_prompt_analysis():
            try:
                res = analyze_prompt_with_gemini_web(temp_image_path, ref_image=use_ref, template_mode=template_mode, keep_hair=keep_hair, keep_acc=keep_acc, keep_bg=keep_bg, model_file=selected_model, user_suggestions=user_suggestions)
                
                # Delete temp file
                if self.temp_image_path:
                    try:
                        os.remove(self.temp_image_path)
                    except:
                        pass
                    
                if res and res.startswith("ERROR:"):
                    self.prompt_error_signal.emit(res)
                else:
                    self.prompt_success_signal.emit(res)
            except Exception as e:
                self.prompt_error_signal.emit(str(e))
                
        threading.Thread(target=run_prompt_analysis, daemon=True).start()
        
    def on_success(self, text):
        if text:
            import re
            # Replace prompt_ref_xxxx_x.png (with or without preceding reference image/photo) with 'reference photo'
            text = re.sub(r'\b(the\s+)?(reference\s+(image|photo)\s+)?prompt_ref_[a-zA-Z0-9_-]+(\.png)?\b', 'reference photo', text, flags=re.IGNORECASE)
            # Remove duplicate consecutive 'reference photo' phrases if they occur
            text = re.sub(r'\b(reference photo\s+)+reference photo\b', 'reference photo', text, flags=re.IGNORECASE)
            text = re.sub(r'\s+([.,;:"\'\])}])', r'\1', text)
            text = re.sub(r'\s{2,}', ' ', text)
        self.analyzing = False
        self.analyze_btn.setText("✨ Phân tích ảnh")
        self.update_analyze_btn_state()
        self.capture_btn.setEnabled(True)
        self.delete_btn.setEnabled(self.image_pixmap is not None)
        self.delete_result_btn.setEnabled(True)
        self.paste_btn.setEnabled(True)
        self.copy_prompt_btn.setEnabled(True)
        self.custom_prompt_input.setEnabled(True)
        
        en_text = ""
        vi_text = ""
        is_json = False
        
        # Check if the text is JSON format (with or without markdown tags)
        try:
            trimmed = text.strip()
            if trimmed.startswith("```"):
                lines = trimmed.split("\n")
                if len(lines) > 2 and (lines[-1].startswith("```") or lines[-1].strip() == ""):
                    if lines[0].startswith("```json") or lines[0].startswith("```"):
                        trimmed = "\n".join(lines[1:-1]).strip()
            parsed = json.loads(trimmed)
            
            # Enforce exact fields from selected model json if template mode is JSON, ref_image is checked, and model is not None
            selected_model = self.model_profile_select.currentText()
            if self.template_select.currentText() == "JSON Chi Tiết" and self.ref_image_checkbox.isChecked() and selected_model != "None":
                mau_path = BASE_DIR / "mau" / selected_model
                if mau_path.exists():
                    try:
                        with open(mau_path, "r", encoding="utf-8") as f:
                            mau_data = json.load(f)
                        
                        # Overwrite subject key attributes (mirror_rules, age, expression, face)
                        if "subject" in parsed and "subject" in mau_data:
                            for key in ["mirror_rules", "age", "expression", "face", "facial_features"]:
                                if key in mau_data["subject"]:
                                    parsed["subject"][key] = mau_data["subject"][key]
                        
                        # Overwrite hair
                        if self.keep_hair_checkbox.isChecked() and "hair" in mau_data:
                            parsed["hair"] = mau_data["hair"]
                            
                        if "body" in mau_data:
                            parsed["body"] = mau_data["body"]
                            
             
                    except Exception as e:
                        print(f"[Error loading/applying mau.json] {e}")
            
            en_text = json.dumps(parsed, indent=2, ensure_ascii=False)
            vi_text = ""
            is_json = True
        except Exception:
            pass
            
        if is_json:
            self.save_json_btn.setEnabled(True)
        else:
            self.save_json_btn.setEnabled(False)
            
        if not is_json:
            if "[PROMPT_EN]" in text and "[PROMPT_VI]" in text:
                parts = text.split("[PROMPT_VI]", 1)
                en_text = parts[0].replace("[PROMPT_EN]", "").strip()
                vi_text = parts[1].strip()
            else:
                en_text = text.strip()
                vi_text = ""
            
        self.prompt_en = en_text
        self.prompt_vi = vi_text
        
        if is_json:
            # HTML escape characters for safety in pre block
            escaped_json = en_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html_result = f"""
            <div style="margin-bottom: 8px;">
                <span style="color: {DRACULA['green']}; font-weight: bold; font-family: monospace; font-size: 11px;">[JSON RESULT]</span>
                <a href="copy_en" style="color: {DRACULA['cyan']}; text-decoration: none; font-size: 12px; margin-left: 6px;">📋 Copy</a><br/>
                <pre style="color: {DRACULA['foreground']}; font-size: 12px; font-family: Consolas, Monaco, monospace; white-space: pre-wrap;">{escaped_json}</pre>
            </div>
            """
        else:
            orig_html = en_text.replace("\n", "<br/>")
            html_result = f"""
            <div style="margin-bottom: 8px;">
                <span style="color: {DRACULA['green']}; font-weight: bold; font-family: monospace; font-size: 11px;">[PROMPT (EN)]</span>
                <a href="copy_en" style="color: {DRACULA['cyan']}; text-decoration: none; font-size: 12px; margin-left: 6px;">📋 Copy</a><br/>
                <span style="color: {DRACULA['foreground']}; font-size: 13px;">{orig_html}</span>
            </div>
            """
            
            if vi_text:
                trans_html = vi_text.replace("\n", "<br/>")
                html_result += f"""
                <hr style="border: 0; border-top: 1px solid rgba(98, 114, 164, 30); margin: 6px 0;"/>
                <div style="margin-bottom: 8px;">
                    <span style="color: {DRACULA['yellow']}; font-weight: bold; font-family: monospace; font-size: 11px;">[PROMPT (VI)]</span>
                    <a href="copy_vi" style="color: {DRACULA['cyan']}; text-decoration: none; font-size: 12px; margin-left: 6px;">📋 Copy</a><br/>
                    <span style="color: {DRACULA['yellow']}; font-size: 13px;">{trans_html}</span>
                </div>
                """
            
        self.result_view.setHtml(html_result)
        self.load_prompt_list_from_json()
        
    def load_prompt_list_from_json(self):
        prompt_json_path = BASE_DIR / "prompt.json"
        if not prompt_json_path.exists():
            self.prompt_list_view.setHtml(f"<div style='color: {DRACULA['comment']}; font-size: 12px;'>File prompt.json không tồn tại ở thư mục ứng dụng.</div>")
            return
            
        try:
            with open(prompt_json_path, "r", encoding="utf-8") as f:
                prompt_items = json.load(f)
                
            html_parts = []
            for idx, item in enumerate(prompt_items):
                name = item.get("name", "Không tên")
                prompt_content = item.get("prompt", "")
                escaped_prompt = prompt_content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
                
                card_html = f"""
                <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom: 8px; background-color: rgba(68, 71, 90, 60); border: 1px solid rgba(98, 114, 164, 40); border-radius: 6px;">
                    <tr>
                        <td style="padding: 10px; vertical-align: top;">
                            <table width="100%" cellpadding="0" cellspacing="0">
                                <tr>
                                    <td style="vertical-align: top; width: 22px;">
                                        <span style="color: {DRACULA['comment']}; font-weight: bold; font-size: 12px; font-family: 'Segoe UI', sans-serif;">{idx + 1}.</span>
                                    </td>
                                    <td style="vertical-align: top;">
                                        <span style="color: {DRACULA['cyan']}; font-weight: bold; font-size: 12px; font-family: 'Segoe UI', sans-serif;">{name}</span>
                                        <div style="margin-top: 4px; color: {DRACULA['foreground']}; font-size: 11px; font-family: 'Segoe UI', sans-serif; line-height: 1.4;">
                                            {escaped_prompt}
                                        </div>
                                    </td>
                                    <td style="vertical-align: middle; text-align: right; width: 75px; padding-left: 10px;">
                                        <table cellpadding="0" cellspacing="0" align="right" style="background-color: {DRACULA['purple']}; border-radius: 4px;">
                                            <tr>
                                                <td style="padding: 4px 8px; text-align: center; vertical-align: middle;">
                                                    <a href="copy_list_idx_{idx}" style="color: white; text-decoration: none; font-size: 11px; font-weight: bold; white-space: nowrap;">📋 Copy</a>
                                                </td>
                                            </tr>
                                        </table>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                </table>
                """
                html_parts.append(card_html)
                
            if not html_parts:
                self.prompt_list_view.setHtml(f"<div style='color: {DRACULA['comment']}; font-size: 12px;'>Danh sách prompt.json trống.</div>")
            else:
                self.prompt_list_view.setHtml("".join(html_parts))
        except Exception as e:
            self.prompt_list_view.setHtml(f"<div style='color: {DRACULA['red']}; font-size: 12px;'>Không thể đọc file prompt.json: {str(e)}</div>")
        
    def on_error(self, err_msg):
        self.analyzing = False
        self.analyze_btn.setText("✨ Phân tích ảnh")
        self.update_analyze_btn_state()
        self.capture_btn.setEnabled(True)
        self.delete_btn.setEnabled(self.image_pixmap is not None)
        self.delete_result_btn.setEnabled(False)
        self.paste_btn.setEnabled(True)
        self.copy_prompt_btn.setEnabled(False)
        self.save_json_btn.setEnabled(False)
        self.custom_prompt_input.setEnabled(True)
        
        self.result_view.setHtml(f"<span style='color: {DRACULA['red']}; font-weight: bold;'>⚠️ Lỗi: {err_msg}</span>")
        
    def handle_link_clicked(self, url):
        link_str = url.toString()
        if link_str == "copy_en":
            self.copy_english_prompt()
        elif link_str == "copy_vi":
            self.copy_vietnamese_prompt()
        elif link_str.startswith("copy_list_idx_"):
            try:
                idx = int(link_str.replace("copy_list_idx_", ""))
                prompt_json_path = BASE_DIR / "prompt.json"
                if prompt_json_path.exists():
                    with open(prompt_json_path, "r", encoding="utf-8") as f:
                        prompt_items = json.load(f)
                    if 0 <= idx < len(prompt_items):
                        clipboard = QApplication.clipboard()
                        clipboard.setText(prompt_items[idx].get("prompt", ""))
                        QToolTip.showText(QCursor.pos(), f"Đã copy prompt '{prompt_items[idx].get('name')}'!")
            except Exception as e:
                print(f"[Error copying from prompt list] {e}")

    def copy_english_prompt(self):
        if hasattr(self, "prompt_en") and self.prompt_en:
            clipboard = QApplication.clipboard()
            clipboard.setText(self.prompt_en)
            QToolTip.showText(QCursor.pos(), "Đã copy Prompt Tiếng Anh vào Clipboard!")

    def copy_vietnamese_prompt(self):
        if hasattr(self, "prompt_vi") and self.prompt_vi:
            clipboard = QApplication.clipboard()
            clipboard.setText(self.prompt_vi)
            QToolTip.showText(QCursor.pos(), "Đã copy Prompt Tiếng Việt vào Clipboard!")

    def copy_prompt(self):
        text_list = []
        if hasattr(self, "prompt_en") and self.prompt_en:
            text_list.append(self.prompt_en)
        if hasattr(self, "prompt_vi") and self.prompt_vi:
            text_list.append(self.prompt_vi)
            
        txt = "\n\n".join(text_list).strip()
        if txt:
            clipboard = QApplication.clipboard()
            clipboard.setText(txt)
            self.copy_prompt_btn.setText("✔️")
            QTimer.singleShot(1500, lambda: self.copy_prompt_btn.setText("📋"))
            QToolTip.showText(QCursor.pos(), "Đã copy vào Clipboard!")

    def save_json_file(self):
        if not hasattr(self, "prompt_en") or not self.prompt_en:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu kết quả JSON",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(self.prompt_en)
                QToolTip.showText(QCursor.pos(), "Đã lưu file JSON thành công!")
            except Exception as e:
                QMessageBox.warning(self, "Lỗi", f"Không thể lưu file: {str(e)}")

    def save_prompt_settings(self):
        if hasattr(self, "refreshing_profiles") and self.refreshing_profiles:
            return
        if self.model_profile_select.count() == 0:
            return
        data = {
            "prompt_ref_image": self.ref_image_checkbox.isChecked(),
            "prompt_ref_bg": self.ref_bg_checkbox.isChecked(),
            "prompt_keep_hair": self.keep_hair_checkbox.isChecked(),
            "prompt_remove_acc": self.keep_acc_checkbox.isChecked(),
            "prompt_model_profile": self.model_profile_select.currentText(),
            "prompt_template_mode": self.template_select.currentText()
        }
        save_settings(data)

    def update_analyze_btn_state(self):
        has_image = self.image_pixmap is not None
        has_input = len(self.custom_prompt_input.toPlainText().strip()) > 0
        self.analyze_btn.setEnabled((has_image or has_input) and not self.analyzing)

    def refresh_model_profiles(self):
        self.refreshing_profiles = True
        current_selection = self.model_profile_select.currentText() if self.model_profile_select.count() > 0 else "None"
        
        # If refreshing for the very first time (during setup), try loading from settings if it exists
        if current_selection == "None" and self.model_profile_select.count() == 0:
            settings = load_settings()
            current_selection = settings.get("prompt_model_profile", "None")

        self.model_profile_select.clear()
        self.model_profile_select.addItem("None")
        
        mau_dir = BASE_DIR / "mau"
        mau_dir.mkdir(exist_ok=True)
        
        for file in mau_dir.glob("*.json"):
            self.model_profile_select.addItem(file.name)
            
        # Restore selection if it still exists
        idx = self.model_profile_select.findText(current_selection)
        if idx >= 0:
            self.model_profile_select.setCurrentIndex(idx)
        else:
            self.model_profile_select.setCurrentText("None")
        self.refreshing_profiles = False

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_model_profiles()
        self.load_prompt_list_from_json()

# ===============================
# OVERLAY WINDOW
# ===============================

# ===============================
# MODERN ICON BUTTON (SVG Rendered)
# ===============================
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush
from PyQt6.QtCore import QPoint, QRect, QRectF, Qt, QByteArray
from PyQt6.QtSvg import QSvgRenderer

SVG_ICONS = {
    "profile": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
  <path fill="url(#geminiGrad)" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 4c1.93 0 3.5 1.57 3.5 3.5S13.93 13 12 13s-3.5-1.57-3.5-3.5S10.07 6 12 6zm0 14c-2.03 0-4.43-1-5.75-2.64C7.74 15.82 10.57 15 12 15s4.26.82 5.75 2.36C16.43 19 14.03 20 12 20z"/>
  <defs>
    <linearGradient id="geminiGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#8ab4f8"/>
      <stop offset="50%" stop-color="#9b8af9"/>
      <stop offset="100%" stop-color="#c58af9"/>
    </linearGradient>
  </defs>
</svg>""",

    "settings": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
  <path fill="{color}" d="M19.43 12.98c.04-.32.07-.64.07-.98s-.03-.66-.07-.98l2.11-1.65c.19-.15.24-.42.12-.64l-2-3.46c-.12-.22-.39-.3-.61-.22l-2.49 1c-.52-.4-1.08-.73-1.69-.98l-.38-2.65C14.46 2.18 14.25 2 14 2h-4c-.25 0-.46.18-.49.42l-.38 2.65c-.61.25-1.17.59-1.69.98l-2.49-1c-.23-.09-.49 0-.61.22l-2 3.46c-.13.22-.07.49.12.64l2.11 1.65c-.04.32-.07.65-.07.98s.03.66.07.98l-2.11 1.65c-.19.15-.24.42-.12.64l2 3.46c.12.22.39.3.61.22l2.49-1c.52.4 1.08.73 1.69.98l.38 2.65c.03.24.24.42.49.42h4c.25 0 .46-.18.49-.42l.38-2.65c.61-.25 1.17-.59 1.69-.98l2.49 1c.23.09.49 0 .61-.22l2-3.46c.12-.22.07-.49-.12-.64l-2.11-1.65zM12 15.5c-1.93 0-3.5-1.57-3.5-3.5s1.57-3.5 3.5-3.5 3.5 1.57 3.5 3.5-1.57 3.5-3.5 3.5z"/>
</svg>""",

    "maximize": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
  <path fill="{color}" d="M18 4H6c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 14H6V6h12v12z"/>
</svg>""",

    "close": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
  <path fill="{color}" d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
</svg>""",

    "context": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
  <path fill="{color}" d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-2 14H7v-2h10v2zm0-4H7v-2h10v2zm0-4H7V7h10v2z"/>
</svg>"""
}

class ModernIconButton(QPushButton):
    def __init__(self, icon_type, parent=None):
        super().__init__(parent)
        self.icon_type = icon_type # "profile", "settings", "maximize", "close"
        self.hovered = False
        self.setFixedSize(28, 28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def enterEvent(self, event):
        self.hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.rect()
        
        # Determine background and border colors based on hover and type
        if self.icon_type == "close" and self.hovered:
            bg_color = QColor(DRACULA['btn_red'])
            border_color = QColor(DRACULA['btn_red'])
            icon_color_str = "white"
        else:
            bg_color = QColor(DRACULA['current_line']) if self.hovered else QColor("transparent")
            if THEME_MODE == "light":
                border_color = QColor(DRACULA['cyan']) if self.hovered else QColor(0, 0, 0, 45)
            else:
                border_color = QColor(DRACULA['cyan']) if self.hovered else QColor(255, 255, 255, 60)
            icon_color_str = DRACULA['foreground']
            
        painter.setBrush(QBrush(bg_color))
        painter.setPen(QPen(border_color, 1))
        
        # Draw circular background border
        painter.drawEllipse(rect.adjusted(1, 1, -1, -1))
        
        # Get raw SVG string and fill in dynamic color
        raw_svg = SVG_ICONS.get(self.icon_type, "")
        if "{color}" in raw_svg:
            raw_svg = raw_svg.format(color=icon_color_str)
            
        renderer = QSvgRenderer(QByteArray(raw_svg.encode('utf-8')))
        padding = 6
        renderer.render(painter, QRectF(rect.adjusted(padding, padding, -padding, -padding)))
        
        painter.end()

class Overlay(QWidget):
    transcription_success = pyqtSignal(str, str, str, str)
    transcription_error = pyqtSignal(str, str)
    transcription_progress = pyqtSignal(str, str)
    ocr_success_signal = pyqtSignal(str)
    ocr_error_signal = pyqtSignal(str)
    ocr_status_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.history = []  # Lưu lịch sử dịch thuật
        self.selected_file_paths = []
        self.file_widgets = {}
        self.file_results = {}
        self.current_transcribing_file_path = None
        self.selected_dir_path = None
        self.ocr_original_text = ""
        self.ocr_translated_text = ""
        self.ocr_ongoing = False
        self.interim_subtitle = None  # (original_draft, translated_draft)
        
        # Download state
        self.is_downloading = False
        self.download_progress = 0.0
        self.download_thread = None
        self.download_queue = []
        self.download_success_count = 0
        self.download_fail_count = 0
        self.first_hide_notification_shown = False
        
        self.init_ui()
        self._drag_pos = None

        # Tải thư mục đã chọn lần trước nếu có
        settings = load_settings()
        last_dir = settings.get("last_dir", "")
        if last_dir and os.path.exists(last_dir):
            self.selected_dir_path = last_dir
            self.refresh_btn.setEnabled(True)
            self.clear_dir_btn.show()
            self.load_files_from_dir(last_dir)

        self.geometry_save_timer = QTimer(self)
        self.geometry_save_timer.setSingleShot(True)
        self.geometry_save_timer.timeout.connect(self.save_geometry_to_settings)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_subtitle)
        self.timer.start(100)
    def init_ui(self):
        self.blink_timer = QTimer(self)
        self.blink_timer.setInterval(500)
        self.blink_state = False
        self.current_led_state = "idle"
        self.blink_timer.timeout.connect(self.blink_status_led)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            |
            Qt.WindowType.WindowStaysOnTopHint
            |
            Qt.WindowType.Tool
        )

        self.setAttribute(
            Qt.WidgetAttribute.WA_TranslucentBackground
        )

        screen = QApplication.primaryScreen()
        geo = screen.geometry()

        width = geo.width()
        height = geo.height()

        self.setMinimumSize(450, 180)
        self.setMouseTracking(True)

        settings = load_settings()
        w = settings.get("window_width", 750)
        h = settings.get("window_height", 420)
        x = settings.get("window_x", 100)
        y = settings.get("window_y", height - 490)
        self.setGeometry(x, y, w, h)

        window_layout = QVBoxLayout()
        window_layout.setContentsMargins(0, 0, 0, 0)
        
        self.container = QFrame()
        self.container.setObjectName("container")
        self.container.setMouseTracking(True)
        
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(15, 12, 15, 15)
        container_layout.setSpacing(10)
        
        title_layout = QHBoxLayout()
        
        logo_label = QLabel()
        logo_path = os.path.join(BASE_DIR, "icon.png")
        if os.path.exists(logo_path):
            logo_pixmap = QPixmap(logo_path)
            logo_label.setPixmap(logo_pixmap.scaled(18, 18, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        title_layout.addWidget(logo_label)
        
        self.title_label = QLabel("Gemini Translate")
        title_layout.addWidget(self.title_label)

        self.status_led = QLabel()
        self.status_led.setFixedSize(8, 8)
        self.status_led.setStyleSheet("background-color: #6C7A9C; border-radius: 4px;")
        title_layout.addWidget(self.status_led)

        self.status_text_label = QLabel("Sẵn sàng")
        title_layout.addWidget(self.status_text_label)

        # Hộp trạng thái kết nối Server Colab (chỉ dùng cho Online Mode)
        self.server_status_widget = QWidget()
        self.server_status_widget.setObjectName("server_status_widget")
        server_layout = QHBoxLayout(self.server_status_widget)
        server_layout.setContentsMargins(8, 2, 8, 2)
        server_layout.setSpacing(6)
        
        # Thêm phong cách capsule cho widget Server
        self.server_status_widget.setStyleSheet("""
            QWidget#server_status_widget {
                background: rgba(255, 255, 255, 0.06);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                padding: 1px;
            }
        """)
        
        self.server_status_led = QLabel()
        self.server_status_led.setFixedSize(8, 8)
        self.server_status_led.setStyleSheet("background-color: #ffa500; border-radius: 4px;") # Màu cam mặc định khi khởi tạo
        
        self.server_status_label = QLabel("Server")
        self.server_status_label.setStyleSheet("font-size: 11px; font-weight: bold; font-family: 'Segoe UI', sans-serif; background: transparent; border: none; color: #ffa500;")
        
        self.server_refresh_btn = QPushButton("↻")
        self.server_refresh_btn.setFixedSize(14, 14)
        self.server_refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.server_refresh_btn.setToolTip("Kiểm tra kết nối Server Colab")
        self.server_refresh_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                font-weight: bold;
                font-size: 12px;
                color: #8ab4f8;
                padding: 0;
                margin: 0;
            }
            QPushButton:hover {
                color: #a3c7ff;
            }
        """)
        self.server_refresh_btn.clicked.connect(self.check_server_connection_async)
        
        server_layout.addWidget(self.server_status_led)
        server_layout.addWidget(self.server_status_label)
        server_layout.addWidget(self.server_refresh_btn)
        title_layout.addWidget(self.server_status_widget)
        
        self.update_server_status_visibility()

        self.lang_switch_widget = QWidget()
        self.lang_switch_widget.setObjectName("lang_switch_widget")
        lang_switch_layout = QHBoxLayout(self.lang_switch_widget)
        lang_switch_layout.setContentsMargins(8, 2, 8, 2)
        lang_switch_layout.setSpacing(6)
        
        self.src_lang_badge = QLabel("EN")
        self.src_lang_badge.setStyleSheet("font-weight: bold; font-family: 'Segoe UI', sans-serif; background: transparent; border: none; font-size: 11px;")
        
        self.swap_lang_btn = QPushButton("⇄")
        self.swap_lang_btn.setFixedSize(16, 16)
        self.swap_lang_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.swap_lang_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                font-weight: bold;
                font-size: 14px;
                color: #8ab4f8;
                padding: 0px;
                margin: 0px;
            }
            QPushButton:hover {
                color: #a3c7ff;
            }
        """)
        self.swap_lang_btn.clicked.connect(self.swap_languages)
        
        self.tgt_lang_badge = QLabel("VI")
        self.tgt_lang_badge.setStyleSheet("font-weight: bold; font-family: 'Segoe UI', sans-serif; background: transparent; border: none; font-size: 11px;")
        
        lang_switch_layout.addWidget(self.src_lang_badge)
        lang_switch_layout.addWidget(self.swap_lang_btn)
        lang_switch_layout.addWidget(self.tgt_lang_badge)
        
        title_layout.addWidget(self.lang_switch_widget)
        self.update_lang_status_ui()

        title_layout.addStretch()

        # Nút Ngữ cảnh 📝
        self.context_btn = ModernIconButton("context", self)
        self.context_btn.setToolTip("Thiết lập ngữ cảnh & Chủ đề dịch")
        self.context_btn.clicked.connect(self.toggle_context_panel)
        title_layout.addWidget(self.context_btn)

        # Nút Profile 👤
        self.profile_btn = ModernIconButton("profile", self)
        self.profile_btn.setToolTip("Kết nối Gemini Web / Chọn Profile")
        self.profile_btn.clicked.connect(self.open_profile_dialog)
        title_layout.addWidget(self.profile_btn)

        # Nút Cài đặt ⚙
        self.settings_btn = ModernIconButton("settings", self)
        self.settings_btn.clicked.connect(self.open_settings_dialog)
        title_layout.addWidget(self.settings_btn)
        
        # Nút phóng to thu nhỏ ⛶
        self.max_btn = ModernIconButton("maximize", self)
        self.max_btn.clicked.connect(self.toggle_maximize)
        title_layout.addWidget(self.max_btn)

        # Nút đóng ×
        self.close_btn = ModernIconButton("close", self)
        self.close_btn.clicked.connect(self.close_app)
        title_layout.addWidget(self.close_btn)
        
        container_layout.addLayout(title_layout)

        # 2. Hàng điều khiển (Bắt đầu nghe & Song ngữ) - Nằm ngay dưới title bar và trên log_view
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(0, 5, 0, 5)
        control_layout.setSpacing(10)
        
        self.listen_btn = QPushButton("Bắt đầu nghe")
        self.listen_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.listen_btn.clicked.connect(self.toggle_listening)
        control_layout.addWidget(self.listen_btn)
        
        self.mode_select = QComboBox()
        self.rebuild_mode_select()
        self.mode_select.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mode_select.currentIndexChanged.connect(self.update_log_display)
        control_layout.addWidget(self.mode_select)
        
        control_layout.addStretch()
        container_layout.addLayout(control_layout)

        # 2.5 Khung bảng điều khiển Ngữ cảnh (Mặc định ẩn)
        self.context_panel = QFrame()
        self.context_panel.setObjectName("context_panel")
        self.context_panel.setVisible(False)
        self.context_panel.setMaximumHeight(180)
        
        context_panel_layout = QVBoxLayout(self.context_panel)
        context_panel_layout.setContentsMargins(10, 8, 10, 8)
        context_panel_layout.setSpacing(6)
        
        # Nhãn tiêu đề
        context_title_layout = QHBoxLayout()
        context_title_lbl = QLabel("Ngữ cảnh & Thuật ngữ cuộc họp")
        context_title_lbl.setStyleSheet("font-weight: bold; font-size: 11px; color: #8ab4f8;")
        context_title_layout.addWidget(context_title_lbl)
        context_title_layout.addStretch()
        
        # Nút đóng bảng ngữ cảnh
        close_panel_btn = QPushButton("✕")
        close_panel_btn.setFixedSize(14, 14)
        close_panel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_panel_btn.setStyleSheet("background: transparent; border: none; color: #ff5555; font-weight: bold; font-size: 10px;")
        close_panel_btn.clicked.connect(self.toggle_context_panel)
        context_title_layout.addWidget(close_panel_btn)
        context_panel_layout.addLayout(context_title_layout)
        
        # Chia đôi trái/phải: trái nhập văn bản gốc, phải chứa prompt kết quả
        context_body_layout = QHBoxLayout()
        context_body_layout.setSpacing(10)
        
        # Cột trái: Nhập mô tả
        left_layout = QVBoxLayout()
        left_layout.setSpacing(4)
        left_lbl = QLabel("1. Mô tả bằng tiếng Việt:")
        left_lbl.setStyleSheet("font-size: 10px; color: #6C7A9C;")
        left_layout.addWidget(left_lbl)
        
        self.context_desc_input = QTextEdit()
        self.context_desc_input.setPlaceholderText("Ví dụ: Buổi meeting nói về dự án Veltrix, báo cáo tiến độ và Go Live ngày 07/07...")
        self.context_desc_input.setStyleSheet("""
            QTextEdit {
                background-color: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 4px;
                color: #e3e3e3;
                font-size: 11px;
                font-family: 'Segoe UI', sans-serif;
            }
        """)
        left_layout.addWidget(self.context_desc_input)
        
        # Nút bấm Phân tích
        self.analyze_context_btn = QPushButton("Phân tích Ngữ cảnh")
        self.analyze_context_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.analyze_context_btn.clicked.connect(self.analyze_meeting_context)
        self.analyze_context_btn.setStyleSheet("""
            QPushButton {
                background-color: #0b57d0;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #1a6fe8;
            }
        """)
        left_layout.addWidget(self.analyze_context_btn)
        context_body_layout.addLayout(left_layout, 1)
        
        # Cột phải: Cấu trúc prompt AI hiểu
        right_layout = QVBoxLayout()
        right_layout.setSpacing(4)
        right_lbl = QLabel("2. AI System Instructions (Có thể chỉnh sửa):")
        right_lbl.setStyleSheet("font-size: 10px; color: #6C7A9C;")
        right_layout.addWidget(right_lbl)
        
        self.context_prompt_output = QTextEdit()
        self.context_prompt_output.setPlaceholderText("Sau khi click 'Phân tích Ngữ cảnh', prompt cấu trúc sẽ tự động tạo ở đây...")
        self.context_prompt_output.setStyleSheet("""
            QTextEdit {
                background-color: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 4px;
                color: #A3BE8C;
                font-size: 10px;
                font-family: monospace;
            }
        """)
        right_layout.addWidget(self.context_prompt_output)
        
        # Nút Áp dụng ngữ cảnh
        self.apply_context_btn = QPushButton("Áp dụng Ngữ cảnh")
        self.apply_context_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.apply_context_btn.clicked.connect(self.apply_meeting_context)
        self.apply_context_btn.setStyleSheet("""
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8ab4f8, stop:1 #c58af9);
                color: #131314;
                border: none;
                border-radius: 4px;
                padding: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #a3c7ff, stop:1 #d8a3ff);
            }
        """)
        right_layout.addWidget(self.apply_context_btn)
        context_body_layout.addLayout(right_layout, 1)
        
        context_panel_layout.addLayout(context_body_layout)
        container_layout.addWidget(self.context_panel)

        # Restore last saved meeting context if present
        settings_tmp = load_settings()
        self.context_desc_input.setPlainText(settings_tmp.get("meeting_context_desc", ""))
        self.context_prompt_output.setPlainText(settings_tmp.get("meeting_context_prompt", ""))

        # 3. Khung hiển thị phụ đề (QTextEdit) đặt trực tiếp ở thân ứng dụng (Nền Glassmorphism phẳng)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.log_view.setHtml(f"<span style='color: {DRACULA['comment']}; font-style: italic;'>Click vào \"Bắt đầu nghe\" để bắt đầu ghi âm và dịch</span>")
        container_layout.addWidget(self.log_view, 1)

        # 4. Hàng nút chức năng ở dưới cùng (Footer bar) - Chỉ chứa Copy và Clear ở góc trái
        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(0, 5, 0, 0)
        footer_layout.setSpacing(6)
        
        self.copy_btn = QPushButton("📋")
        self.copy_btn.setToolTip("Sao chép lịch sử phụ đề")
        self.copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.copy_btn.clicked.connect(self.copy_to_clipboard)
        footer_layout.addWidget(self.copy_btn)
        
        self.clear_tab_btn = QPushButton("🗑️")
        self.clear_tab_btn.setToolTip("Xóa lịch sử phụ đề")
        self.clear_tab_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_tab_btn.clicked.connect(self.clear_active_tab_history)
        footer_layout.addWidget(self.clear_tab_btn)
        
        footer_layout.addStretch()
        container_layout.addLayout(footer_layout)
        
        window_layout.addWidget(self.container)
        self.setLayout(window_layout)

        # Kích hoạt hover tracking để đổi con trỏ chuột khi đến viền
        self.setMouseTracking(True)
        self.container.setMouseTracking(True)

        # Tạo nút co giãn kích thước ở góc dưới bên phải
        self.sizegrip = QSizeGrip(self)
        self.sizegrip.setStyleSheet("QSizeGrip { width: 16px; height: 16px; background: transparent; }")
        
        self.apply_theme_and_font()

    def apply_theme_and_font(self):
        global THEME_MODE, DRACULA
        
        # 1. Update Dracula theme color definitions based on Theme
        if THEME_MODE == "light":
            # Gemini Light Palette
            DRACULA["background"] = "#F0F4F9"
            DRACULA["background_alpha"] = "rgba(240, 244, 249, 230)"
            DRACULA["current_line"] = "#FFFFFF"
            DRACULA["foreground"] = "#1F1F1F"
            DRACULA["comment"] = "#5F6368"
            DRACULA["cyan"] = "#0B57D0" # Gemini Royal Blue
            DRACULA["purple"] = "#0B57D0"
            DRACULA["yellow"] = "#1F1F1F" # Charcoal for source text
            DRACULA["green"] = "#0B57D0" # Blue for target translation text
            DRACULA["green_rgb"] = "11, 87, 208"
        else:
            # Slate Premium Dark Palette
            DRACULA["background"] = "#1A1C23"
            DRACULA["background_alpha"] = "rgba(26, 28, 35, 230)"
            DRACULA["current_line"] = "#2B2D3A"
            DRACULA["foreground"] = "#E5E9F0"
            DRACULA["comment"] = "#6C7A9C"
            DRACULA["cyan"] = "#5E81AC"
            DRACULA["purple"] = "#88C0D0"
            DRACULA["yellow"] = "#EBCB8B"
            DRACULA["green"] = "#A3BE8C"
            DRACULA["green_rgb"] = "163, 190, 140"
            
        # 2. Stylesheet adjustments
        if THEME_MODE == "light":
            # Container Light styles
            if IS_LISTENING:
                # Glowing border when active
                self.container.setStyleSheet(f"""
                    QFrame#container {{
                        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #F0F4F9, stop:0.5 #E2EAF4, stop:1 #E8F0FE);
                        border: 2px solid qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0B57D0, stop:0.5 #9b8af9, stop:1 #c58af9);
                        border-radius: 12px;
                    }}
                """)
            else:
                self.container.setStyleSheet(f"""
                    QFrame#container {{
                        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #F0F4F9, stop:0.5 #E2EAF4, stop:1 #E8F0FE);
                        border: 1px solid rgba(0, 0, 0, 0.12);
                        border-radius: 12px;
                    }}
                """)
                
            self.title_label.setStyleSheet("color: #1F1F1F; font-size: 13px; font-weight: bold; font-family: 'Segoe UI', sans-serif; background: transparent;")
            self.status_text_label.setStyleSheet("color: #5F6368; font-size: 11px; font-family: 'Segoe UI', sans-serif; background: transparent;")
            
            # Switcher badge style Light
            self.lang_switch_widget.setStyleSheet("""
                QWidget#lang_switch_widget {
                    background-color: rgba(0, 0, 0, 0.04);
                    border: 1px solid rgba(0, 0, 0, 0.08);
                    border-radius: 12px;
                }
            """)
            self.src_lang_badge.setStyleSheet("color: #1F1F1F; font-weight: bold; font-family: 'Segoe UI', sans-serif; background: transparent; border: none; font-size: 11px;")
            self.tgt_lang_badge.setStyleSheet("color: #1F1F1F; font-weight: bold; font-family: 'Segoe UI', sans-serif; background: transparent; border: none; font-size: 11px;")
            self.swap_lang_btn.setStyleSheet("""
                QPushButton {
                    background: transparent;
                    border: none;
                    font-weight: bold;
                    font-size: 14px;
                    color: #0b57d0;
                    padding: 0px;
                    margin: 0px;
                }
                QPushButton:hover {
                    color: #1a73e8;
                }
            """)
            
            # Log View Light (Glassmorphic look)
            self.log_view.setStyleSheet(f"""
                QTextEdit {{
                    color: #1F1F1F;
                    background-color: rgba(255, 255, 255, 0.65);
                    border: 1px solid rgba(0, 0, 0, 0.08);
                    border-radius: 12px;
                    padding: 14px;
                }}
                QScrollBar:vertical {{
                    border: none;
                    background: rgba(0, 0, 0, 10);
                    width: 6px;
                    border-radius: 3px;
                }}
                QScrollBar::handle:vertical {{
                    background: rgba(0, 0, 0, 60);
                    border-radius: 3px;
                    min-height: 20px;
                }}
            """)
            
            # Buttons Light (Copy, Clear)
            btn_style = """
                QPushButton {
                    color: #1F1F1F;
                    background-color: #ffffff;
                    border: 1px solid rgba(0, 0, 0, 0.12);
                    border-radius: 14px;
                    min-width: 28px;
                    min-height: 28px;
                    max-width: 28px;
                    max-height: 28px;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background-color: #e2eaf4;
                    border-color: #0b57d0;
                }
            """
            self.copy_btn.setStyleSheet(btn_style)
            self.clear_tab_btn.setStyleSheet(btn_style)
            
            # Listen Button Light
            if IS_LISTENING:
                self.listen_btn.setText("Tạm dừng nghe")
                self.listen_btn.setStyleSheet("""
                    QPushButton {
                        color: white;
                        background-color: #BF616A;
                        border: none;
                        border-radius: 6px;
                        padding: 8px 30px;
                        font-weight: bold;
                        font-family: 'Segoe UI', sans-serif;
                        font-size: 13px;
                    }
                    QPushButton:hover {
                        background-color: #C9717A;
                    }
                """)
            else:
                self.listen_btn.setText("Bắt đầu nghe")
                self.listen_btn.setStyleSheet("""
                    QPushButton {
                        color: white;
                        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0B57D0, stop:1 #9b8af9);
                        border: none;
                        border-radius: 6px;
                        padding: 8px 30px;
                        font-weight: bold;
                        font-family: 'Segoe UI', sans-serif;
                        font-size: 13px;
                    }
                    QPushButton:hover {
                        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1a73e8, stop:1 #b3a3ff);
                    }
                """)
                
            # Mode select Light
            self.mode_select.setStyleSheet("""
                QComboBox {
                    color: #1F1F1F;
                    background-color: #ffffff;
                    border: 1px solid rgba(0, 0, 0, 0.12);
                    border-radius: 14px;
                    padding: 4px 20px 4px 12px;
                    font-size: 12px;
                    font-family: 'Segoe UI', sans-serif;
                }
                QComboBox::drop-down {
                    border: none;
                    width: 16px;
                }
                QComboBox QAbstractItemView {
                    color: #1F1F1F;
                    background-color: #ffffff;
                    selection-background-color: #e2eaf4;
                    border: 1px solid rgba(0, 0, 0, 0.1);
                    border-radius: 8px;
                }
            """)
        else:
            # Container Dark styles
            if IS_LISTENING:
                # Glowing border when active
                self.container.setStyleSheet(f"""
                    QFrame#container {{
                        background-color: {DRACULA['background_alpha']};
                        border: 2px solid qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8ab4f8, stop:0.5 #9b8af9, stop:1 #c58af9);
                        border-radius: 12px;
                    }}
                """)
            else:
                self.container.setStyleSheet(f"""
                    QFrame#container {{
                        background-color: {DRACULA['background_alpha']};
                        border: 1px solid rgba(255, 255, 255, 0.08);
                        border-radius: 12px;
                    }}
                """)
                
            self.title_label.setStyleSheet(f"color: {DRACULA['foreground']}; font-size: 13px; font-weight: bold; font-family: 'Segoe UI', sans-serif; background: transparent;")
            self.status_text_label.setStyleSheet(f"color: {DRACULA['comment']}; font-size: 11px; font-family: 'Segoe UI', sans-serif; background: transparent;")
            
            # Switcher badge style Dark
            self.lang_switch_widget.setStyleSheet("""
                QWidget#lang_switch_widget {
                    background-color: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    border-radius: 12px;
                }
            """)
            self.src_lang_badge.setStyleSheet(f"color: {DRACULA['foreground']}; font-weight: bold; font-family: 'Segoe UI', sans-serif; background: transparent; border: none; font-size: 11px;")
            self.tgt_lang_badge.setStyleSheet(f"color: {DRACULA['foreground']}; font-weight: bold; font-family: 'Segoe UI', sans-serif; background: transparent; border: none; font-size: 11px;")
            self.swap_lang_btn.setStyleSheet("""
                QPushButton {
                    background: transparent;
                    border: none;
                    font-weight: bold;
                    font-size: 14px;
                    color: #8ab4f8;
                    padding: 0px;
                    margin: 0px;
                }
                QPushButton:hover {
                    color: #a3c7ff;
                }
            """)
            
            # Log View Dark
            self.log_view.setStyleSheet(f"""
                QTextEdit {{
                    color: {DRACULA['foreground']};
                    background-color: rgba(255, 255, 255, 0.04);
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    border-radius: 12px;
                    padding: 14px;
                }}
                QScrollBar:vertical {{
                    border: none;
                    background: rgba(0, 0, 0, 30);
                    width: 6px;
                    border-radius: 3px;
                }}
                QScrollBar::handle:vertical {{
                    background: {DRACULA['comment']};
                    border-radius: 3px;
                    min-height: 20px;
                }}
            """)
            
            # Buttons Dark (Copy, Clear)
            btn_style = f"""
                QPushButton {{
                    color: {DRACULA['foreground']};
                    background-color: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    border-radius: 14px;
                    min-width: 28px;
                    min-height: 28px;
                    max-width: 28px;
                    max-height: 28px;
                    font-size: 13px;
                }}
                QPushButton:hover {{
                    background-color: rgba(255, 255, 255, 0.12);
                    border-color: {DRACULA['cyan']};
                }}
            """
            self.copy_btn.setStyleSheet(btn_style)
            self.clear_tab_btn.setStyleSheet(btn_style)
            
            # Listen Button Dark
            if IS_LISTENING:
                self.listen_btn.setText("Tạm dừng nghe")
                self.listen_btn.setStyleSheet("""
                    QPushButton {
                        color: white;
                        background-color: #BF616A;
                        border: none;
                        border-radius: 6px;
                        padding: 8px 30px;
                        font-weight: bold;
                        font-family: 'Segoe UI', sans-serif;
                        font-size: 13px;
                    }
                    QPushButton:hover {
                        background-color: #C9717A;
                    }
                """)
            else:
                self.listen_btn.setText("Bắt đầu nghe")
                self.listen_btn.setStyleSheet(f"""
                    QPushButton {{
                        color: #131314;
                        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8ab4f8, stop:1 #c58af9);
                        border: none;
                        border-radius: 6px;
                        padding: 8px 30px;
                        font-weight: bold;
                        font-family: 'Segoe UI', sans-serif;
                        font-size: 13px;
                    }}
                    QPushButton:hover {{
                        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #a3c7ff, stop:1 #d8a3ff);
                    }}
                """)
                
            # Mode select Dark
            self.mode_select.setStyleSheet(f"""
                QComboBox {{
                    color: {DRACULA['foreground']};
                    background-color: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    border-radius: 14px;
                    padding: 4px 20px 4px 12px;
                    font-size: 12px;
                    font-family: 'Segoe UI', sans-serif;
                }}
                QComboBox::drop-down {{
                    border: none;
                    width: 16px;
                }}
                QComboBox QAbstractItemView {{
                    color: {DRACULA['foreground']};
                    background-color: {DRACULA['background']};
                    selection-background-color: {DRACULA['current_line']};
                    border: 1px solid rgba(255, 255, 255, 0.12);
                    border-radius: 8px;
                }}
            """)
            
        # Update top-right vector buttons icons to trigger color refresh
        self.profile_btn.update()
        self.settings_btn.update()
        self.max_btn.update()
        self.close_btn.update()

        # Update server status capsule style
        if hasattr(self, "server_status_widget"):
            if THEME_MODE == "light":
                self.server_status_widget.setStyleSheet("""
                    QWidget#server_status_widget {
                        background: rgba(0, 0, 0, 0.04);
                        border: 1px solid rgba(0, 0, 0, 0.08);
                        border-radius: 10px;
                        padding: 1px;
                    }
                """)
            else:
                self.server_status_widget.setStyleSheet("""
                    QWidget#server_status_widget {
                        background: rgba(255, 255, 255, 0.06);
                        border: 1px solid rgba(255, 255, 255, 0.1);
                        border-radius: 10px;
                        padding: 1px;
                    }
                """)

        # Update context panel styling dynamically
        if hasattr(self, "context_panel"):
            if THEME_MODE == "light":
                self.context_panel.setStyleSheet("""
                    QFrame#context_panel {
                        background-color: rgba(0, 0, 0, 0.02);
                        border: 1px solid rgba(0, 0, 0, 0.06);
                        border-radius: 8px;
                    }
                """)
                self.context_desc_input.setStyleSheet("""
                    QTextEdit {
                        background-color: #ffffff;
                        border: 1px solid rgba(0, 0, 0, 0.12);
                        border-radius: 4px;
                        color: #1F1F1F;
                        font-size: 11px;
                        font-family: 'Segoe UI', sans-serif;
                    }
                """)
                self.context_prompt_output.setStyleSheet("""
                    QTextEdit {
                        background-color: #ffffff;
                        border: 1px solid rgba(0, 0, 0, 0.12);
                        border-radius: 4px;
                        color: #4A9F5E;
                        font-size: 10px;
                        font-family: monospace;
                    }
                """)
            else:
                self.context_panel.setStyleSheet("""
                    QFrame#context_panel {
                        background-color: rgba(255, 255, 255, 0.02);
                        border: 1px solid rgba(255, 255, 255, 0.05);
                        border-radius: 8px;
                    }
                """)
                self.context_desc_input.setStyleSheet("""
                    QTextEdit {
                        background-color: rgba(0, 0, 0, 0.2);
                        border: 1px solid rgba(255, 255, 255, 0.08);
                        border-radius: 4px;
                        color: #e3e3e3;
                        font-size: 11px;
                        font-family: 'Segoe UI', sans-serif;
                    }
                """)
                self.context_prompt_output.setStyleSheet("""
                    QTextEdit {
                        background-color: rgba(0, 0, 0, 0.2);
                        border: 1px solid rgba(255, 255, 255, 0.08);
                        border-radius: 4px;
                        color: #A3BE8C;
                        font-size: 10px;
                        font-family: monospace;
                    }
                """)

    def set_status_state(self, state: str, text: str = None):
        self.current_led_state = state
        self.blink_state = True
        
        if state == "listening":
            self.blink_timer.start()
            self.status_led.setStyleSheet("background-color: #A3BE8C; border-radius: 4px;")
            if text:
                self.status_text_label.setText(text)
            else:
                self.status_text_label.setText("Đang ghi âm...")
            self.status_text_label.setStyleSheet(f"color: {DRACULA['green']}; font-size: 11px; font-family: 'Segoe UI', sans-serif; background: transparent;")
        elif state == "error":
            self.blink_timer.start()
            self.status_led.setStyleSheet("background-color: #BF616A; border-radius: 4px;")
            if text:
                self.status_text_label.setText(text)
            else:
                self.status_text_label.setText("Lỗi")
            self.status_text_label.setStyleSheet(f"color: {DRACULA['red']}; font-size: 11px; font-weight: bold; font-family: 'Segoe UI', sans-serif; background: transparent;")
        else: # idle
            self.blink_timer.stop()
            self.status_led.setStyleSheet("background-color: #6C7A9C; border-radius: 4px;")
            if text:
                self.status_text_label.setText(text)
            else:
                self.status_text_label.setText("Sẵn sàng")
            self.status_text_label.setStyleSheet(f"color: {DRACULA['comment']}; font-size: 11px; font-family: 'Segoe UI', sans-serif; background: transparent;")

    def blink_status_led(self):
        self.blink_state = not self.blink_state
        if self.current_led_state == "listening":
            color = "#A3BE8C" if self.blink_state else "#4C566A"
        elif self.current_led_state == "error":
            color = "#BF616A" if self.blink_state else "#4C566A"
        else:
            color = "#6C7A9C"
            self.blink_timer.stop()
            
        self.status_led.setStyleSheet(f"background-color: {color}; border-radius: 4px;")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.sizegrip.move(self.width() - self.sizegrip.width() - 4, self.height() - self.sizegrip.height() - 4)
        if hasattr(self, "geometry_save_timer"):
            self.geometry_save_timer.start(1000)

    def moveEvent(self, event):
        super().moveEvent(event)
        if hasattr(self, "geometry_save_timer"):
            self.geometry_save_timer.start(1000)

    def save_geometry_to_settings(self):
        if self.isMaximized() or self.isMinimized():
            return
        geo = self.geometry()
        save_settings({
            "window_width": geo.width(),
            "window_height": geo.height(),
            "window_x": geo.x(),
            "window_y": geo.y()
        })

    def show_app(self):
        self.show()
        self.raise_()
        self.activateWindow()
        if hasattr(self, "active_settings_dialog") and self.active_settings_dialog:
            self.active_settings_dialog.show()
            self.active_settings_dialog.raise_()
            self.active_settings_dialog.activateWindow()
        if hasattr(self, "active_profile_dialog") and self.active_profile_dialog:
            self.active_profile_dialog.show()
            self.active_profile_dialog.raise_()
            self.active_profile_dialog.activateWindow()
            
        # Hide the floating button
        if hasattr(self, "floating_btn") and self.floating_btn:
            self.floating_btn.hide()

    def hide_app(self):
        self.hide()
        if hasattr(self, "active_settings_dialog") and self.active_settings_dialog:
            self.active_settings_dialog.hide()
        if hasattr(self, "active_profile_dialog") and self.active_profile_dialog:
            self.active_profile_dialog.hide()
            
        # Show the floating button
        if hasattr(self, "floating_btn") and self.floating_btn:
            self.floating_btn.show()

    def quit_app(self):
        # Close the floating button
        if hasattr(self, "floating_btn") and self.floating_btn:
            self.floating_btn.close()
        self.tray_icon.hide()
        QApplication.quit()

    def toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
            self.max_btn.setText("⛶")
        else:
            self.showMaximized()
            self.max_btn.setText("❐")

    def close_app(self):
        self.hide_app()
        self.show_tray_message()

    def closeEvent(self, event):
        event.ignore()
        self.hide_app()
        self.show_tray_message()

    def on_tray_icon_activated(self, reason):
        if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick):
            if self.isVisible():
                self.hide_app()
            else:
                self.show_app()

    def show_tray_message(self):
        if not self.first_hide_notification_shown:
            self.tray_icon.showMessage(
                "Dịch thuật",
                "Ứng dụng vẫn đang chạy ẩn dưới thanh taskbar (System Tray). Nhấp vào icon để hiện lại.",
                QSystemTrayIcon.MessageIcon.Information,
                3000
            )
            self.first_hide_notification_shown = True

    def on_tab_changed(self, index):
        if index == 0:
            self.listen_btn.show()
            self.copy_btn.show()
            self.model_select.show()
            self.mode_select.show()
            self.clear_tab_btn.show()
        elif index == 1:
            self.listen_btn.hide()
            self.copy_btn.hide()
            self.model_select.show()
            self.mode_select.show()
            self.clear_tab_btn.show()
        else:
            self.listen_btn.hide()
            self.copy_btn.hide()
            self.model_select.hide()
            self.mode_select.hide()
            self.clear_tab_btn.hide()

    def paste_ocr_image(self):
        if getattr(self, "ocr_ongoing", False):
            return
            
        clipboard = QApplication.clipboard()
        mime_data = clipboard.mimeData()
        pixmap = None
        
        # 1. Clipboard chứa dữ liệu ảnh trực tiếp
        if mime_data.hasImage():
            image = clipboard.image()
            pixmap = QPixmap.fromImage(image)
            
        # 2. Clipboard chứa file ảnh được copy từ thư mục local
        elif mime_data.hasUrls():
            urls = mime_data.urls()
            if urls:
                file_path = urls[0].toLocalFile()
                if file_path and os.path.exists(file_path):
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"]:
                        pixmap = QPixmap(file_path)
                        
        if pixmap is None or pixmap.isNull():
            QMessageBox.information(
                self, 
                "Thông báo", 
                "Không tìm thấy ảnh hoặc tệp ảnh hợp lệ trong Clipboard. Vui lòng copy ảnh hoặc file ảnh trước."
            )
            return
            
        self.ocr_ongoing = True
        self.ocr_captured_pixmap = pixmap
        
        # Cập nhật hiển thị ảnh xem trước
        scaled_pixmap = pixmap.scaled(
            140, 140,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.ocr_img_label.setPixmap(scaled_pixmap)
        self.ocr_zoom_btn.show()
        self.ocr_copy_img_btn.show()
        self.ocr_copy_img_overlay_btn.show()
        self.ocr_save_img_btn.show()
        self.ocr_save_img_overlay_btn.show()
        self.ocr_refresh_img_btn.hide()
        self.ocr_img_container.show()
        
        # Lưu ra file tạm và chạy OCR
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"ocr_capture_{os.getpid()}.png")
        pixmap.save(temp_file_path, "PNG")
        
        translate_option = self.ocr_translate_select.currentText()
        translate_flag = (translate_option != "None")
        
        self.execute_ocr(temp_file_path, translate_flag)

    def start_ocr_selection(self):
        self.tabs.setCurrentIndex(2)
        if getattr(self, "ocr_ongoing", False):
            return
        if self.current_transcribing_file_path is not None:
            self.ocr_result_view.setHtml(
                f"<span style='color: {DRACULA['orange']}; font-weight: bold;'>⚠️ Hệ thống đang bận dịch file. Vui lòng đợi file dịch xong.</span>"
            )
            return
            
        self.ocr_ongoing = True
        # Vô hiệu hóa nút ngay lập tức để tránh người dùng click liên tiếp (spam)
        self.ocr_btn.setEnabled(False)
        self.ocr_btn.setText("⏳ Chuẩn bị...")
        self.ocr_paste_btn.setEnabled(False)
        
        self.hide()
        if hasattr(self, "active_settings_dialog") and self.active_settings_dialog:
            self.active_settings_dialog.hide()
        if hasattr(self, "active_profile_dialog") and self.active_profile_dialog:
            self.active_profile_dialog.hide()
        self.ocr_copy_src_btn.hide()
        self.ocr_copy_tgt_btn.hide()
        self.ocr_copy_img_btn.hide()
        self.ocr_copy_img_overlay_btn.hide()
        self.ocr_save_img_btn.hide()
        self.ocr_save_img_overlay_btn.hide()
        self.ocr_zoom_btn.hide()
        self.ocr_refresh_img_btn.hide()
        self.ocr_img_container.hide()
        QTimer.singleShot(250, self.open_screen_selector)

    def open_screen_selector(self):
        self.selector = ScreenSelector()
        self.selector.region_selected.connect(self.on_ocr_region_selected)
        
        # Khi đóng trình chụp ảnh, nếu cuộc gọi OCR không chạy (huỷ thao tác), ta cần khôi phục lại nút
        def on_selector_closed():
            self.show_app()
            if not getattr(self.selector, "selection_made", False):
                self.ocr_btn.setEnabled(True)
                self.ocr_btn.setText("📷 Quét vùng màn hình (OCR)")
                self.ocr_paste_btn.setEnabled(True)
                self.ocr_ongoing = False
                
        self.selector.closed.connect(on_selector_closed)
        self.selector.start_selection()

    def on_ocr_region_selected(self, x, y, w, h):
        self.show_app()
        
        screen = QGuiApplication.screenAt(QPoint(x, y))
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        if not screen:
            self.ocr_result_view.setHtml(f"<span style='color: {DRACULA['red']}; font-weight: bold;'>⚠️ Lỗi: Không tìm thấy màn hình.</span>")
            self.ocr_btn.setEnabled(True)
            self.ocr_btn.setText("📷 Quét vùng màn hình (OCR)")
            self.ocr_paste_btn.setEnabled(True)
            self.ocr_ongoing = False
            return
            
        local_x = x - screen.geometry().x()
        local_y = y - screen.geometry().y()
        pixmap = screen.grabWindow(0, local_x, local_y, w, h)
        if pixmap.isNull():
            self.ocr_result_view.setHtml(f"<span style='color: {DRACULA['red']}; font-weight: bold;'>⚠️ Lỗi: Không thể chụp ảnh vùng đã chọn.</span>")
            self.ocr_btn.setEnabled(True)
            self.ocr_btn.setText("📷 Quét vùng màn hình (OCR)")
            self.ocr_paste_btn.setEnabled(True)
            self.ocr_ongoing = False
            return
            
        self.ocr_captured_pixmap = pixmap

        # Show preview immediately after capture
        scaled_pixmap = pixmap.scaled(
            140, 140,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.ocr_img_label.setPixmap(scaled_pixmap)
        self.ocr_zoom_btn.show()
        self.ocr_copy_img_btn.show()
        self.ocr_copy_img_overlay_btn.show()
        self.ocr_save_img_btn.show()
        self.ocr_save_img_overlay_btn.show()
        self.ocr_refresh_img_btn.hide() # Nút làm mới chỉ hiện sau khi quét xong
        self.ocr_img_container.show()

        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"ocr_capture_{os.getpid()}.png")
        pixmap.save(temp_file_path, "PNG")
        
        translate_option = self.ocr_translate_select.currentText()
        translate_flag = (translate_option != "None")
        
        self.execute_ocr(temp_file_path, translate_flag)

    def re_run_ocr(self):
        if not hasattr(self, "ocr_captured_pixmap") or self.ocr_captured_pixmap is None or self.ocr_captured_pixmap.isNull():
            return
        if getattr(self, "ocr_ongoing", False):
            return
            
        self.ocr_ongoing = True
        
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"ocr_capture_{os.getpid()}.png")
        self.ocr_captured_pixmap.save(temp_file_path, "PNG")
        
        translate_option = self.ocr_translate_select.currentText()
        translate_flag = (translate_option != "None")
        
        self.execute_ocr(temp_file_path, translate_flag)

    def execute_ocr(self, temp_file_path, translate_flag):
        if not translate_flag:
            status_html = f"""
                <div style="color: {DRACULA['cyan']}; font-weight: bold; font-size: 13px;">
                    ⏳ Đang nhận diện chữ... Vui lòng đợi.
                </div>
            """
        else:
            status_html = f"""
                <div style="color: {DRACULA['cyan']}; font-weight: bold; font-size: 13px;">
                    ⏳ Đang nhận diện chữ và dịch... Vui lòng đợi.
                </div>
            """
        self.ocr_result_view.setHtml(status_html)
        self.ocr_btn.setEnabled(False)
        self.ocr_btn.setText("⏳ Đang quét...")
        self.ocr_paste_btn.setEnabled(False)
        
        # Giữ các nút Copy/Lưu Ảnh và Zoom Ảnh luôn hiện, ẩn nút Làm mới khi đang quét
        self.ocr_zoom_btn.show()
        self.ocr_copy_img_btn.show()
        self.ocr_copy_img_overlay_btn.show()
        self.ocr_save_img_btn.show()
        self.ocr_save_img_overlay_btn.show()
        self.ocr_refresh_img_btn.hide()
        
        def run_ocr():
            try:
                res_text = None
                selected_mode = self.ocr_mode_select.currentText()
                
                # Option A: Gemini Web OCR
                if "Gemini Web" in selected_mode:
                    if USE_GEMINI and gemini_chat:
                        try:
                            print("[OCR] Đang dùng Gemini Web...")
                            raw_res = ocr_with_gemini_web(temp_file_path, translate=translate_flag)
                            if raw_res and not raw_res.startswith("ERROR:"):
                                if not translate_flag:
                                    if "[TRANSLATED]" in raw_res:
                                        raw_res = raw_res.split("[TRANSLATED]", 1)[0]
                                    cleaned = raw_res.replace("[ORIGINAL]", "").strip()
                                    res_text = f"[ORIGINAL]\n{cleaned}\n[TRANSLATED]\n"
                                else:
                                    res_text = raw_res
                            else:
                                res_text = raw_res
                        except Exception as gemini_err:
                            print(f"[OCR Warning] Lỗi Gemini Web OCR: {gemini_err}.")
                            res_text = f"ERROR: Lỗi Gemini Web: {str(gemini_err)}"
                    else:
                        res_text = "ERROR: Chưa kết nối Gemini Web. Vui lòng kết nối Profile ở nút hình người 👤 hoặc chọn chế độ Tesseract OCR (Offline)."
                
                # Option B: Tesseract OCR
                elif "Tesseract OCR" in selected_mode:
                    if ocr_processor:
                        print("[OCR] Đang dùng Tesseract OCR...")
                        try:
                            extracted_text = ocr_processor.run_local_ocr(temp_file_path)
                            if extracted_text:
                                if translate_flag:
                                    print(f"[OCR] Nhận dạng thành công. Tiến hành dịch...")
                                    translated_text = translate_text(extracted_text).strip()
                                    res_text = f"[ORIGINAL]\n{extracted_text}\n[TRANSLATED]\n{translated_text}"
                                else:
                                    res_text = f"[ORIGINAL]\n{extracted_text}\n[TRANSLATED]\n"
                            else:
                                if translate_flag:
                                    res_text = "[ORIGINAL]\nKhông tìm thấy chữ trong vùng chọn.\n[TRANSLATED]\nNo text found in selected region."
                                else:
                                    res_text = "[ORIGINAL]\nKhông tìm thấy chữ trong vùng chọn.\n[TRANSLATED]\n"
                        except Exception as ocr_err:
                            res_text = f"ERROR: {str(ocr_err)}"
                    else:
                        res_text = "ERROR: Tesseract OCR module chưa được khởi tạo thành công."
                
                # Option C: EasyOCR
                elif "EasyOCR" in selected_mode:
                    if OCRReader:
                        print("[OCR] Đang dùng EasyOCR...")
                        try:
                            global easyocr_reader
                            if easyocr_reader is None:
                                print("[OCR] Khởi tạo EasyOCR Reader...")
                                easyocr_reader = OCRReader(status_callback=lambda msg: self.ocr_status_signal.emit(msg))
                            
                            extracted_text = easyocr_reader.extract_text(temp_file_path)
                            if extracted_text:
                                if translate_flag:
                                    print(f"[OCR] Nhận dạng thành công. Tiến hành dịch...")
                                    translated_text = translate_text(extracted_text).strip()
                                    res_text = f"[ORIGINAL]\n{extracted_text}\n[TRANSLATED]\n{translated_text}"
                                else:
                                    res_text = f"[ORIGINAL]\n{extracted_text}\n[TRANSLATED]\n"
                            else:
                                if translate_flag:
                                    res_text = "[ORIGINAL]\nKhông tìm thấy chữ trong vùng chọn.\n[TRANSLATED]\nNo text found in selected region."
                                else:
                                    res_text = "[ORIGINAL]\nKhông tìm thấy chữ trong vùng chọn.\n[TRANSLATED]\n"
                        except Exception as ocr_err:
                            res_text = f"ERROR: {str(ocr_err)}"
                    else:
                        res_text = "ERROR: EasyOCR module chưa được khởi tạo thành công."
                
                try:
                    os.remove(temp_file_path)
                except:
                    pass
                self.ocr_success_signal.emit(res_text)
            except Exception as e:
                self.ocr_error_signal.emit(str(e))
                
        threading.Thread(target=run_ocr, daemon=True).start()

    def show_ocr_success(self, text):
        self.ocr_ongoing = False
        self.ocr_btn.setEnabled(True)
        self.ocr_btn.setText("📷 Quét vùng màn hình (OCR)")
        self.ocr_paste_btn.setEnabled(True)
        
        if text.startswith("ERROR:"):
            self.ocr_result_view.setHtml(f"<span style='color: {DRACULA['red']}; font-weight: bold;'>{text}</span>")
            self.ocr_img_container.hide()
            self.ocr_zoom_btn.hide()
            self.ocr_copy_img_btn.hide()
            self.ocr_copy_img_overlay_btn.hide()
            self.ocr_save_img_btn.hide()
            self.ocr_save_img_overlay_btn.hide()
            self.ocr_refresh_img_btn.hide()
            return
            
        # Tách lấy văn bản gốc và bản dịch từ Gemini Web
        
        original_text = ""
        translated_text = ""
        
        if "[ORIGINAL]" in text and "[TRANSLATED]" in text:
            parts = text.split("[TRANSLATED]", 1)
            original_text = parts[0].replace("[ORIGINAL]", "").strip()
            translated_text = parts[1].strip()
        else:
            translated_text = text.strip()
            original_text = "Không phân tách được văn bản gốc từ Gemini."
            
        self.ocr_original_text = original_text
        self.ocr_translated_text = translated_text
        
        # Cập nhật hiển thị ảnh xem trước khi quét thành công
        if hasattr(self, "ocr_captured_pixmap") and self.ocr_captured_pixmap is not None:
            scaled_pixmap = self.ocr_captured_pixmap.scaled(
                140, 140, 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.ocr_img_label.setPixmap(scaled_pixmap)
            self.ocr_zoom_btn.show()
            self.ocr_copy_img_btn.show()
            self.ocr_copy_img_overlay_btn.show()
            self.ocr_save_img_btn.show()
            self.ocr_save_img_overlay_btn.show()
            self.ocr_refresh_img_btn.show()
            self.ocr_img_container.show()
        
        # Format HTML hiển thị đẹp đẽ không kèm theo nhãn thừa
        orig_html = original_text.replace("\n", "<br/>")
        
        if translated_text:
            trans_html = translated_text.replace("\n", "<br/>")
            html_result = f"""
            <div style="margin-bottom: 8px;">
                <span style="color: {DRACULA['cyan']}; font-weight: bold; font-family: monospace; font-size: 11px;">[Nguồn]:</span>
                <a href="copy_src" style="color: {DRACULA['comment']}; text-decoration: none; font-size: 12px; margin-left: 6px;">📋</a><br/>
                <span style="color: {DRACULA['foreground']}; font-size: 13px;">{orig_html}</span>
            </div>
            <hr style="border: 0; border-top: 1px solid rgba(98, 114, 164, 30); margin: 6px 0;"/>
            <div style="margin-bottom: 8px;">
                <span style="color: {DRACULA['green']}; font-weight: bold; font-family: monospace; font-size: 11px;">[Bản dịch]:</span>
                <a href="copy_tgt" style="color: {DRACULA['comment']}; text-decoration: none; font-size: 12px; margin-left: 6px;">📋</a><br/>
                <span style="color: {DRACULA['yellow']}; font-size: 14px; font-weight: bold;">{trans_html}</span>
            </div>
            """
            self.ocr_result_view.setHtml(html_result)
            
            # Hiện hai nút copy text
            self.ocr_copy_src_btn.show()
            self.ocr_copy_tgt_btn.show()
        else:
            html_result = f"""
            <div style="margin-bottom: 8px;">
                <span style="color: {DRACULA['cyan']}; font-weight: bold; font-family: monospace; font-size: 11px;">[Nguồn]:</span>
                <a href="copy_src" style="color: {DRACULA['comment']}; text-decoration: none; font-size: 12px; margin-left: 6px;">📋</a><br/>
                <span style="color: {DRACULA['foreground']}; font-size: 13px;">{orig_html}</span>
            </div>
            
            """
            self.ocr_result_view.setHtml(html_result)
            
            # Hiện nút copy nguồn và ẩn nút copy đích
            self.ocr_copy_src_btn.show()
            self.ocr_copy_tgt_btn.hide()

    def show_ocr_status(self, status_msg):
        status_html = f"""
            <div style="color: {DRACULA['cyan']}; font-weight: bold; font-size: 13px;">
                ⏳ {status_msg}
            </div>
        """
        self.ocr_result_view.setHtml(status_html)

    def show_ocr_error(self, err_msg):
        self.ocr_ongoing = False
        self.ocr_btn.setEnabled(True)
        self.ocr_btn.setText("📷 Quét vùng màn hình (OCR)")
        self.ocr_paste_btn.setEnabled(True)
        self.ocr_result_view.setHtml(f"<span style='color: {DRACULA['red']}; font-weight: bold;'>⚠️ Lỗi: {err_msg}</span>")
        self.ocr_copy_src_btn.hide()
        self.ocr_copy_tgt_btn.hide()
        self.ocr_copy_img_btn.hide()
        self.ocr_copy_img_overlay_btn.hide()
        self.ocr_save_img_btn.hide()
        self.ocr_save_img_overlay_btn.hide()
        self.ocr_img_container.hide()
        self.ocr_zoom_btn.hide()
        self.ocr_refresh_img_btn.hide()

    def handle_ocr_link_clicked(self, url):
        link_str = url.toString()
        if link_str == "copy_src":
            self.copy_ocr_src_result()
        elif link_str == "copy_tgt":
            self.copy_ocr_tgt_result()

    def copy_ocr_src_result(self):
        if self.ocr_original_text:
            clipboard = QApplication.clipboard()
            clipboard.setText(self.ocr_original_text)
            self.ocr_copy_src_btn.setText("Đã copy!")
            QTimer.singleShot(1500, lambda: self.ocr_copy_src_btn.setText("Copy Nguồn"))
            QToolTip.showText(QCursor.pos(), "Đã copy văn bản nguồn!")

    def copy_ocr_tgt_result(self):
        if self.ocr_translated_text:
            clipboard = QApplication.clipboard()
            clipboard.setText(self.ocr_translated_text)
            self.ocr_copy_tgt_btn.setText("Đã copy!")
            QTimer.singleShot(1500, lambda: self.ocr_copy_tgt_btn.setText("Copy Đích"))
            QToolTip.showText(QCursor.pos(), "Đã copy văn bản dịch!")

    def zoom_ocr_image(self):
        if hasattr(self, "ocr_captured_pixmap") and self.ocr_captured_pixmap is not None:
            dialog = ImageZoomDialog(self.ocr_captured_pixmap, self)
            dialog.exec()

    def copy_ocr_image_result(self):
        if hasattr(self, "ocr_captured_pixmap") and self.ocr_captured_pixmap is not None:
            clipboard = QApplication.clipboard()
            clipboard.setPixmap(self.ocr_captured_pixmap)
            self.ocr_copy_img_btn.setText("Đã copy!")
            QTimer.singleShot(1500, lambda: self.ocr_copy_img_btn.setText("Copy Ảnh"))
            
            self.ocr_copy_img_overlay_btn.setText("✅")
            self.ocr_copy_img_overlay_btn.setToolTip("Đã copy ảnh!")
            QTimer.singleShot(1500, lambda: (self.ocr_copy_img_overlay_btn.setText("📋"), self.ocr_copy_img_overlay_btn.setToolTip("Copy ảnh vào Clipboard")))

    def save_ocr_image_result(self):
        if hasattr(self, "ocr_captured_pixmap") and self.ocr_captured_pixmap is not None and not self.ocr_captured_pixmap.isNull():
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Lưu ảnh quét OCR",
                os.path.join(os.path.expanduser("~"), "Downloads", "ocr_capture.png"),
                "Image Files (*.png *.jpg *.bmp)"
            )
            if file_path:
                try:
                    self.ocr_captured_pixmap.save(file_path)
                    self.ocr_save_img_btn.setText("Đã lưu!")
                    QTimer.singleShot(1500, lambda: self.ocr_save_img_btn.setText("Lưu Ảnh"))
                    
                    self.ocr_save_img_overlay_btn.setText("✅")
                    self.ocr_save_img_overlay_btn.setToolTip("Đã lưu ảnh!")
                    QTimer.singleShot(1500, lambda: (self.ocr_save_img_overlay_btn.setText("💾"), self.ocr_save_img_overlay_btn.setToolTip("Tải ảnh về máy")))
                except Exception as e:
                    QMessageBox.warning(self, "Lỗi", f"Không thể lưu ảnh: {e}")

    def format_file_size(self, size_in_bytes):
        if size_in_bytes < 1024:
            return f"{size_in_bytes} B"
        elif size_in_bytes < 1024 * 1024:
            return f"{size_in_bytes / 1024:.2f} KB"
        elif size_in_bytes < 1024 * 1024 * 1024:
            return f"{size_in_bytes / (1024 * 1024):.2f} MB"
        else:
            return f"{size_in_bytes / (1024 * 1024 * 1024):.2f} GB"

    def on_files_selected(self, file_paths, append=False):
        if append:
            # Lọc các file đã có trong danh sách
            new_paths = [fp for fp in file_paths if fp not in self.selected_file_paths]
            if not new_paths:
                return
            self.selected_file_paths.extend(new_paths)
            paths_to_process = new_paths
        else:
            self.selected_file_paths = file_paths
            
            # Clear previous items in scroll area
            while self.file_list_layout.count() > 0:
                item = self.file_list_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                    
            self.file_widgets = {}
            self.file_results = {}
            self.current_transcribing_file_path = None
            if self.selected_dir_path is None:
                save_settings({"last_dir": ""})
                self.clear_dir_btn.hide()
            self.result_view.clear()
            paths_to_process = file_paths
        
        if not self.selected_file_paths:
            self.file_label.setText("Chưa chọn file")
            self.file_label.setToolTip("")
            self.file_list_scroll.hide()
            return
            
        total_size = 0
        for fp in self.selected_file_paths:
            try:
                total_size += os.path.getsize(fp)
            except Exception:
                pass
        total_size_str = self.format_file_size(total_size)
        self.file_label.setText(f"📁 Đã chọn {len(self.selected_file_paths)} file ({total_size_str})")
        self.file_label.setToolTip("\n".join(self.selected_file_paths))
        
        # Populate scroll area for new paths
        for fp in paths_to_process:
            filename = os.path.basename(fp)
            try:
                size_bytes = os.path.getsize(fp)
                size_str = self.format_file_size(size_bytes)
            except Exception:
                size_str = "N/A"
                
            row_widget = QWidget()
            row_widget.setObjectName("file_row")
            row_widget.setStyleSheet(f"""
                QWidget#file_row {{
                    background-color: transparent;
                    border: 1px solid rgba(98, 114, 164, 40);
                    border-radius: 4px;
                }}
                QWidget#file_row:hover {{
                    background-color: rgba(68, 71, 90, 80);
                    border-color: {DRACULA['cyan']};
                }}
            """)
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(6, 4, 6, 4)
            row_layout.setSpacing(8)
            
            lbl = QLabel(f"📄 {filename} ({size_str})")
            lbl.setToolTip(fp)
            lbl.setStyleSheet(f"""
                color: {DRACULA['foreground']};
                font-size: 11px;
                font-family: 'Segoe UI', sans-serif;
                background: transparent;
            """)
            row_layout.addWidget(lbl, 1)
            
            btn = QPushButton("Bắt đầu")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton {{
                    color: white;
                    background-color: rgba({DRACULA['btn_green_rgb']}, 160);
                    border: 1px solid rgba({DRACULA['btn_green_rgb']}, 50);
                    border-radius: 4px;
                    padding: 2px 10px;
                    font-size: 10px;
                    font-weight: bold;
                    font-family: 'Segoe UI', sans-serif;
                }}
                QPushButton:hover {{
                    background-color: rgba({DRACULA['btn_green_rgb']}, 220);
                }}
                QPushButton:disabled {{
                    background-color: rgba(68, 71, 90, 50);
                    color: rgba(248, 248, 242, 60);
                    border: 1px solid rgba(98, 114, 164, 30);
                }}
            """)
            btn.clicked.connect(lambda checked, path=fp: self.start_single_file_transcription(path))
            row_layout.addWidget(btn)
            
            del_btn = QPushButton("Xóa")
            del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            del_btn.setStyleSheet(f"""
                QPushButton {{
                    color: {DRACULA['foreground']};
                    background-color: rgba(255, 85, 85, 120);
                    border: 1px solid rgba(255, 85, 85, 50);
                    border-radius: 4px;
                    padding: 2px 8px;
                    font-size: 10px;
                    font-weight: bold;
                    font-family: 'Segoe UI', sans-serif;
                }}
                QPushButton:hover {{
                    background-color: rgba(255, 85, 85, 200);
                }}
                QPushButton:disabled {{
                    background-color: rgba(68, 71, 90, 50);
                    color: rgba(248, 248, 242, 60);
                    border: 1px solid rgba(98, 114, 164, 30);
                }}
            """)
            del_btn.clicked.connect(lambda checked, path=fp: self.remove_single_file(path))
            row_layout.addWidget(del_btn)
            
            self.file_widgets[fp] = {
                'button': btn,
                'del_button': del_btn,
                'label': lbl,
                'row': row_widget
            }
            
            if append:
                # Chèn trước phần stretch (nằm cuối)
                idx = self.file_list_layout.count() - 1
                if idx < 0:
                    idx = 0
                self.file_list_layout.insertWidget(idx, row_widget)
            else:
                self.file_list_layout.addWidget(row_widget)
            
        if not append:
            self.file_list_layout.addStretch()
            
        self.file_list_scroll.show()
        self.update_result_view()

    def browse_file(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Chọn file âm thanh",
            "",
            "Audio Files (*.wav *.mp3 *.m4a *.flac *.ogg);;All Files (*)"
        )
        if file_paths:
            self.on_files_selected(file_paths, append=True)

    def on_files_dropped(self, file_paths):
        self.on_files_selected(file_paths, append=True)

    def browse_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Chọn thư mục chứa file âm thanh", "")
        if dir_path:
            self.selected_dir_path = dir_path
            self.refresh_btn.setEnabled(True)
            self.clear_dir_btn.show()
            save_settings({"last_dir": dir_path})
            self.load_files_from_dir(dir_path)

    def refresh_directory(self):
        if hasattr(self, "selected_dir_path") and self.selected_dir_path:
            if os.path.exists(self.selected_dir_path):
                self.load_files_from_dir(self.selected_dir_path)
                self.result_view.append(f"\n<span style='color: {DRACULA['green']}; font-style: italic;'>[Hệ thống]: Đã làm mới danh sách file từ thư mục.</span>")
            else:
                QMessageBox.warning(self, "Cảnh báo", "Thư mục không còn tồn tại!")
                self.refresh_btn.setEnabled(False)
                self.selected_dir_path = None

    def load_files_from_dir(self, dir_path):
        import glob
        extensions = ["*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg", "*.WAV", "*.MP3", "*.M4A", "*.FLAC", "*.OGG"]
        file_paths = []
        for ext in extensions:
            pattern = os.path.join(dir_path, ext)
            file_paths.extend(glob.glob(pattern))
            
        file_paths = sorted(list(set(file_paths)))
        
        self.on_files_selected(file_paths)
        
        if file_paths:
            total_size = sum(os.path.getsize(fp) for fp in file_paths if os.path.exists(fp))
            total_size_str = self.format_file_size(total_size)
            dir_name = os.path.basename(dir_path)
            self.file_label.setText(f"📁 Thư mục: {dir_name} | Đã tìm thấy {len(file_paths)} file ({total_size_str})")

    def clear_file_tab(self):
        # Clear layout items
        while self.file_list_layout.count() > 0:
            item = self.file_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
                
        self.selected_file_paths = []
        self.file_widgets = {}
        self.file_results = {}
        self.current_transcribing_file_path = None
        self.selected_dir_path = None
        self.refresh_btn.setEnabled(False)
        self.clear_dir_btn.hide()
        save_settings({"last_dir": ""})
        
        self.file_list_scroll.hide()
        self.file_label.setText("Chưa chọn file")
        self.file_label.setToolTip("")
        self.result_view.clear()

    def remove_single_file(self, file_path):
        if not file_path:
            return
            
        if file_path in self.selected_file_paths:
            self.selected_file_paths.remove(file_path)
            
        widgets = self.file_widgets.get(file_path)
        if widgets:
            row_widget = widgets['row']
            self.file_list_layout.removeWidget(row_widget)
            row_widget.deleteLater()
            
        self.file_widgets.pop(file_path, None)
        self.file_results.pop(file_path, None)
        
        if not self.selected_file_paths:
            self.file_label.setText("Chưa chọn file")
            self.file_label.setToolTip("")
            self.file_list_scroll.hide()
            self.result_view.clear()
            return
            
        total_size = 0
        for fp in self.selected_file_paths:
            try:
                total_size += os.path.getsize(fp)
            except Exception:
                pass
        total_size_str = self.format_file_size(total_size)
        self.file_label.setText(f"📁 Đã chọn {len(self.selected_file_paths)} file ({total_size_str})")
        self.file_label.setToolTip("\n".join(self.selected_file_paths))
        
        self.update_result_view()

    def copy_file_result(self):
        text = self.result_view.toPlainText()
        if text:
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            self.result_view.append("\n<span style='color: #4CAF50; font-style: italic;'>[Hệ thống]: Đã copy kết quả vào Clipboard.</span>")

    def enable_all_file_buttons(self):
        for fp, widgets in self.file_widgets.items():
            widgets['button'].setEnabled(True)
            if 'del_button' in widgets:
                widgets['del_button'].setEnabled(True)
        self.clear_tab_btn.setEnabled(True)
        self.drop_area.setEnabled(True)

    def disable_all_file_buttons(self):
        for fp, widgets in self.file_widgets.items():
            widgets['button'].setEnabled(False)
            if 'del_button' in widgets:
                widgets['del_button'].setEnabled(False)
        self.clear_tab_btn.setEnabled(False)
        self.drop_area.setEnabled(False)

    def update_result_view(self):
        final_html = ""
        for fp in self.selected_file_paths:
            res_html = self.file_results.get(fp)
            if res_html:
                final_html += res_html
            else:
                filename = os.path.basename(fp)
                if fp == self.current_transcribing_file_path:
                    final_html += f"""
                    <div style="margin-bottom: 12px; border: 1px dashed rgba(41, 182, 246, 50); border-radius: 6px; padding: 8px; background-color: rgba(41, 182, 246, 5);">
                        <div style="color: {DRACULA['cyan']}; font-weight: bold; font-size: 11px; margin-bottom: 4px;">⌛ Đang dịch: {filename}...</div>
                    </div>
                    """
                else:
                    final_html += f"""
                    <div style="margin-bottom: 12px; border: 1px solid rgba(98, 114, 164, 30); border-radius: 6px; padding: 8px; background-color: rgba(40, 42, 54, 40);">
                        <div style="color: {DRACULA['comment']}; font-size: 11px; margin-bottom: 4px;">📁 {filename} (Chưa dịch)</div>
                    </div>
                    """
        self.result_view.setHtml(final_html)

    def start_single_file_transcription(self, file_path):
        if not file_path or self.current_transcribing_file_path is not None:
            return
        if getattr(self, "ocr_ongoing", False):
            self.result_view.append(f"\n<span style='color: {DRACULA['orange']}; font-style: italic;'>[Hệ thống]: Đang quét OCR. Vui lòng đợi quét xong rồi dịch file.</span>")
            return
            
        self.current_transcribing_file_path = file_path
        self.disable_all_file_buttons()
        
        widgets = self.file_widgets.get(file_path)
        if widgets:
            widgets['button'].setEnabled(False)
            widgets['button'].setText("⏳ Đang dịch...")
            widgets['label'].setStyleSheet(f"color: {DRACULA['cyan']}; font-family: 'Segoe UI', sans-serif; font-size: 11px; font-weight: bold; background: transparent;")
            if 'copy_src_btn' in widgets:
                widgets['copy_src_btn'].hide()
            if 'copy_tgt_btn' in widgets:
                widgets['copy_tgt_btn'].hide()
            
        self.update_result_view()
        
        mode = WHISPER_MODE
        model_name = LOCAL_MODEL_NAME
        api_url = API_URL
        src_lang = SOURCE_LANG
        tgt_lang = TARGET_LANG
        
        print(f"[File STT] Bắt đầu chép lời file đơn lẻ: {file_path}", flush=True)
        
        def run():
            try:
                print(f"[File STT Thread] Đang chạy với mode={mode}, api_url={api_url}", flush=True)
                
                # 1. Transcribe
                if mode == "online":
                    print(f"[File STT Thread] Gửi request POST tới {api_url}...", flush=True)
                
                original_text, _ = transcriber.transcribe(
                    file_path,
                    mode=mode,
                    src_lang=src_lang,
                    local_model_name=model_name,
                    api_url=api_url,
                    gemini_chat=gemini_chat
                )
                
                print(f"[File STT Thread] Nhận dạng thành công. Độ dài text: {len(original_text) if original_text else 0}", flush=True)
                if not original_text:
                    raise Exception("Không thể nhận diện được giọng nói trong file âm thanh này.")
                
                # 2. Translate
                print(f"[File STT Thread] Đang tiến hành dịch thuật...", flush=True)
                translated_text = translate_text(original_text)
                print(f"[File STT Thread] Dịch hoàn tất. Đang gửi tín hiệu về giao diện...", flush=True)
                
                src_code = LANGUAGES.get(src_lang, {}).get("google", "auto").split("-")[0].upper()
                if src_code == "AUTO":
                    src_code = "Auto"
                tgt_code = LANGUAGES.get(tgt_lang, {}).get("google", "vi").split("-")[0].upper()
                
                result_html = f"""
                <div style="margin-bottom: 12px; border: 1px solid rgba(98, 114, 164, 40); border-radius: 6px; padding: 8px; background-color: rgba(40, 42, 54, 80);">
                    <div style="color: {DRACULA['cyan']}; font-weight: bold; font-size: 11px; margin-bottom: 4px;">📁 {os.path.basename(file_path)}</div>
                    <div>
                        <span style="color: {DRACULA['comment']}; font-weight: bold; font-family: monospace; font-size: 11px;">[{src_code} - Nguồn]:</span><br/>
                        <span style="color: {DRACULA['foreground']}; font-size: 13px;">{original_text}</span>
                    </div>
                    <hr style="border: 0; border-top: 1px solid rgba(98, 114, 164, 30); margin: 6px 0;"/>
                    <div>
                        <span style="color: {DRACULA['cyan']}; font-weight: bold; font-family: monospace; font-size: 11px;">[{tgt_code} - Bản dịch]:</span><br/>
                        <span style="color: {DRACULA['yellow']}; font-size: 14px; font-weight: bold;">{translated_text}</span>
                    </div>
                </div>
                """
                self.transcription_success.emit(file_path, result_html, original_text, translated_text)
            except Exception as e:
                err_msg = str(e)
                print(f"[File STT Thread Exception] File: {os.path.basename(file_path)}, Lỗi: {err_msg}", flush=True)
                self.transcription_error.emit(file_path, err_msg)
                
        threading.Thread(target=run, daemon=True).start()

    def copy_text_to_clipboard(self, btn, text, original_label):
        if text:
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            btn.setText("Đã copy!")
            QTimer.singleShot(1500, lambda: btn.setText(original_label))

    def show_transcription_success(self, file_path, result_html, original_text, translated_text):
        self.file_results[file_path] = result_html
        
        widgets = self.file_widgets.get(file_path)
        if widgets:
            btn = widgets['button']
            btn.setText("Dịch lại")
            widgets['label'].setStyleSheet(f"color: {DRACULA['green']}; font-family: 'Segoe UI', sans-serif; font-size: 11px; font-weight: bold; background: transparent;")
            
            # Tạo hoặc hiển thị nút Copy Nguồn và Copy Đích
            has_no_translation = (LANGUAGES.get(TARGET_LANG, {}).get("google") == "none")
            
            # Setup/show copy source button
            if 'copy_src_btn' in widgets:
                widgets['copy_src_btn'].show()
                try:
                    widgets['copy_src_btn'].clicked.disconnect()
                except TypeError:
                    pass
                widgets['copy_src_btn'].clicked.connect(lambda checked, b=widgets['copy_src_btn'], t=original_text: self.copy_text_to_clipboard(b, t, "Copy Nguồn"))
            else:
                copy_src_btn = QPushButton("Copy Nguồn")
                copy_src_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                copy_src_btn.setStyleSheet(f"""
                    QPushButton {{
                        color: {DRACULA['foreground']};
                        background-color: rgba(41, 182, 246, 120);
                        border: 1px solid rgba(41, 182, 246, 50);
                        border-radius: 4px;
                        padding: 2px 8px;
                        font-size: 10px;
                        font-weight: bold;
                        font-family: 'Segoe UI', sans-serif;
                    }}
                    QPushButton:hover {{
                        background-color: rgba(41, 182, 246, 200);
                    }}
                """)
                copy_src_btn.clicked.connect(lambda checked, b=copy_src_btn, t=original_text: self.copy_text_to_clipboard(b, t, "Copy Nguồn"))
                row_layout = widgets['row'].layout()
                idx = row_layout.indexOf(btn)
                if idx != -1:
                    row_layout.insertWidget(idx, copy_src_btn)
                else:
                    row_layout.addWidget(copy_src_btn)
                widgets['copy_src_btn'] = copy_src_btn

            # Setup/show/hide copy translation button
            if not has_no_translation:
                if 'copy_tgt_btn' in widgets:
                    widgets['copy_tgt_btn'].show()
                    try:
                        widgets['copy_tgt_btn'].clicked.disconnect()
                    except TypeError:
                        pass
                    widgets['copy_tgt_btn'].clicked.connect(lambda checked, b=widgets['copy_tgt_btn'], t=translated_text: self.copy_text_to_clipboard(b, t, "Copy Đích"))
                else:
                    copy_tgt_btn = QPushButton("Copy Đích")
                    copy_tgt_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    copy_tgt_btn.setStyleSheet(f"""
                        QPushButton {{
                            color: white;
                            background-color: rgba(241, 250, 140, 120);
                            border: 1px solid rgba(241, 250, 140, 50);
                            border-radius: 4px;
                            padding: 2px 8px;
                            font-size: 10px;
                            font-weight: bold;
                            font-family: 'Segoe UI', sans-serif;
                        }}
                        QPushButton:hover {{
                            background-color: rgba(241, 250, 140, 200);
                        }}
                    """)
                    copy_tgt_btn.clicked.connect(lambda checked, b=copy_tgt_btn, t=translated_text: self.copy_text_to_clipboard(b, t, "Copy Đích"))
                    row_layout = widgets['row'].layout()
                    idx = row_layout.indexOf(btn)
                    if idx != -1:
                        row_layout.insertWidget(idx + 1, copy_tgt_btn)
                    else:
                        row_layout.addWidget(copy_tgt_btn)
                    widgets['copy_tgt_btn'] = copy_tgt_btn
            else:
                if 'copy_tgt_btn' in widgets:
                    widgets['copy_tgt_btn'].hide()
                
        self.current_transcribing_file_path = None
        self.enable_all_file_buttons()
        self.update_result_view()
 
    def show_transcription_progress(self, file_path, progress_msg):
        widgets = self.file_widgets.get(file_path)
        if widgets:
            widgets['button'].setText(progress_msg)
 
    def show_transcription_error(self, file_path, error_msg):
        err_html = f"""
        <div style="margin-bottom: 12px; border: 1px solid rgba(255, 85, 85, 40); border-radius: 6px; padding: 8px; background-color: rgba(255, 85, 85, 8);">
            <div style="color: {DRACULA['red']}; font-weight: bold; font-size: 11px; margin-bottom: 4px;">📁 {os.path.basename(file_path)}</div>
            <div>
                <span style="color: {DRACULA['orange']}; font-size: 13px; font-weight: bold;">⚠️ Lỗi chép lời: {error_msg}</span>
            </div>
        </div>
        """
        self.file_results[file_path] = err_html
        
        widgets = self.file_widgets.get(file_path)
        if widgets:
            btn = widgets['button']
            btn.setText("Dịch lại")
            widgets['label'].setStyleSheet(f"color: {DRACULA['red']}; font-family: 'Segoe UI', sans-serif; font-size: 11px; font-weight: bold; background: transparent;")
            if 'copy_src_btn' in widgets:
                widgets['copy_src_btn'].hide()
            if 'copy_tgt_btn' in widgets:
                widgets['copy_tgt_btn'].hide()
            
        self.current_transcribing_file_path = None
        self.enable_all_file_buttons()
        self.update_result_view()

    def show_transcription_progress(self, file_path, progress_msg):
        widgets = self.file_widgets.get(file_path)
        if widgets:
            widgets['button'].setText(progress_msg)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            rect = self.rect()
            
            # Khởi tạo các cờ xác định vị trí biên co giãn (8 hướng)
            self._resize_top = pos.y() <= 8
            self._resize_bottom = (rect.bottom() - pos.y()) <= 8
            self._resize_left = pos.x() <= 8
            self._resize_right = (rect.right() - pos.x()) <= 8
            
            if self._resize_top or self._resize_bottom or self._resize_left or self._resize_right:
                self._resize_start_pos = event.globalPosition().toPoint()
                self._resize_start_geometry = self.geometry()
            else:
                # Kéo di chuyển cửa sổ khi click vào phần tiêu đề (y <= 45)
                if pos.y() <= 45:
                    self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                else:
                    self._drag_pos = None
            event.accept()

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        rect = self.rect()
        
        # 1. Thay đổi con trỏ chuột khi rê qua các viền/góc
        if event.buttons() == Qt.MouseButton.NoButton:
            t = pos.y() <= 8
            b = (rect.bottom() - pos.y()) <= 8
            l = pos.x() <= 8
            r = (rect.right() - pos.x()) <= 8
            
            if (t and l) or (b and r):
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)  # Góc chéo chính (NW-SE)
            elif (t and r) or (b and l):
                self.setCursor(Qt.CursorShape.SizeBDiagCursor)  # Góc chéo phụ (NE-SW)
            elif t or b:
                self.setCursor(Qt.CursorShape.SizeVerCursor)    # Biên trên/dưới (N-S)
            elif l or r:
                self.setCursor(Qt.CursorShape.SizeHorCursor)    # Biên trái/phải (W-E)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
                
        # 2. Xử lý co giãn hoặc di chuyển khi nhấn giữ chuột trái
        elif event.buttons() == Qt.MouseButton.LeftButton:
            if hasattr(self, "_resize_start_pos"):
                delta = event.globalPosition().toPoint() - self._resize_start_pos
                geom = self._resize_start_geometry
                
                new_x = geom.x()
                new_y = geom.y()
                new_width = geom.width()
                new_height = geom.height()
                
                # Co giãn biên phải / trái
                if self._resize_right:
                    new_width = max(self.minimumWidth(), geom.width() + delta.x())
                elif self._resize_left:
                    diff = delta.x()
                    if geom.width() - diff >= self.minimumWidth():
                        new_x = geom.x() + diff
                        new_width = geom.width() - diff
                        
                # Co giãn biên dưới / trên
                if self._resize_bottom:
                    new_height = max(self.minimumHeight(), geom.height() + delta.y())
                elif self._resize_top:
                    diff = delta.y()
                    if geom.height() - diff >= self.minimumHeight():
                        new_y = geom.y() + diff
                        new_height = geom.height() - diff
                        
                self.setGeometry(new_x, new_y, new_width, new_height)
            elif self._drag_pos is not None:
                self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        if hasattr(self, "_resize_start_pos"):
            del self._resize_start_pos
            del self._resize_start_geometry
        self.setCursor(Qt.CursorShape.ArrowCursor)
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            if pos.y() <= 45:
                self.toggle_maximize()
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        super().keyPressEvent(event)

    def toggle_listening(self):
        global IS_LISTENING
        IS_LISTENING = not IS_LISTENING
        
        if IS_LISTENING:
            self.set_status_state("listening")
        else:
            self.set_status_state("idle")
            
        self.apply_theme_and_font()
        self.update_log_display()

    def open_profile_dialog(self):
        self.profile_btn.setEnabled(False)
        # Mở hộp thoại cấu hình profile Gemini
        active_profile = load_gemini_profile()
        dialog = GeminiProfileDialog(self, active_profile)
        self.active_profile_dialog = dialog
        result = dialog.exec()
        
        if result == QDialog.DialogCode.Accepted:
            profile_name = dialog.get_selected_profile()
            if profile_name and profile_name != "<Không sử dụng>":
                global active_gemini_profile
                if active_gemini_profile != profile_name:
                    def connect_bg():
                        subtitle_queue.put(f"[SYSTEM]\nCONNECTING:{profile_name}")
                        ok, msg = init_gemini_client(profile_name)
                        if ok:
                            subtitle_queue.put(f"[SYSTEM]\nCONNECTED:{profile_name}")
                        else:
                            if "used by another process" in msg or "PermissionError" in msg or "WinError 32" in msg:
                                subtitle_queue.put(f"[SYSTEM]\nFAILED_LOCK:{profile_name}")
                            elif "cookie" in msg.lower() or "đăng nhập" in msg.lower():
                                subtitle_queue.put(f"[SYSTEM]\nFAILED_LOGIN:{profile_name}")
                            else:
                                subtitle_queue.put(f"[SYSTEM]\nFAILED:{profile_name}")
                    threading.Thread(target=connect_bg, daemon=True).start()
                else:
                    self.history.append(("[SYSTEM]", f"Đã chọn active profile: {profile_name}"))
            else:
                self.history.append(("[SYSTEM]", "Đã xóa profile. Dịch thuật quay lại chế độ cũ (Google dịch)."))
                subtitle_queue.put("[SYSTEM]\nFAILED:")
            self.update_log_display()
        self.active_profile_dialog = None
        self.profile_btn.setEnabled(True)

    def rebuild_mode_select(self):
        # Lưu lại index hiện tại để khôi phục nếu được
        curr_idx = self.mode_select.currentIndex()
        if curr_idx < 0:
            curr_idx = 0
            
        src_name = LANGUAGES.get(SOURCE_LANG, {}).get("name_vi", "tiếng Anh").capitalize()
        
        # Ngắt tạm thời tín hiệu để tránh gọi update_log_display liên tiếp khi clear/add
        self.mode_select.blockSignals(True)
        self.mode_select.clear()
        
        if LANGUAGES.get(TARGET_LANG, {}).get("google") == "none":
            self.mode_select.addItems([
                f"Chỉ {src_name}", 
                f"Chỉ {src_name} 1 dòng"
            ])
        else:
            tgt_name = LANGUAGES.get(TARGET_LANG, {}).get("name_vi", "tiếng Việt").capitalize()
            self.mode_select.addItems([
                "Song ngữ", 
                f"Chỉ {src_name}", 
                f"Chỉ {tgt_name}", 
                "Song ngữ 1 dòng", 
                f"Chỉ {src_name} 1 dòng", 
                f"Chỉ {tgt_name} 1 dòng"
            ])
        
        # Khôi phục index cũ nếu hợp lệ
        if curr_idx < self.mode_select.count():
            self.mode_select.setCurrentIndex(curr_idx)
        else:
            self.mode_select.setCurrentIndex(0)
        self.mode_select.blockSignals(False)

    def open_settings_dialog(self):
        self.settings_btn.setEnabled(False)
        dialog = SettingsDialog(self)
        self.active_settings_dialog = dialog
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.history.append(("[SYSTEM]", f"Đã cấu hình dịch: {SOURCE_LANG} ➔ {TARGET_LANG}"))
            self.reload_settings_and_ui()
        self.active_settings_dialog = None
        self.settings_btn.setEnabled(True)

    def clear_history(self):
        self.history.clear()
        self.update_log_display()

    def clear_active_tab_history(self):
        self.clear_history()


    def copy_to_clipboard(self):
        if not self.history:
            return
        
        # Build text to copy
        text_lines = []
        for en, vi in self.history:
            if en == "[SYSTEM]":
                text_lines.append(f"[HỆ THỐNG]: {vi}")
            else:
                text_lines.append(f"[ENG]: {en}")
                text_lines.append(f"[VIE]: {vi}")
                text_lines.append("-" * 40)
        
        full_text = "\n".join(text_lines)
        
        clipboard = QApplication.clipboard()
        clipboard.setText(full_text)
        
        # Show a quick system notice to the user that it has been copied
        self.history.append(("[SYSTEM]", "Đã sao chép lịch sử dịch thuật vào Clipboard."))
        self.update_log_display()

    def change_whisper_mode(self):
        global WHISPER_MODE, LOCAL_MODEL_NAME, model
        selected = self.model_select.currentText()
        if selected == "Online":
            WHISPER_MODE = "online"
            self.history.append(("[SYSTEM]", "Đã chuyển sang chế độ Online (Tối ưu nhẹ)."))
            self.update_log_display()
        elif selected == "Gemini Web":
            WHISPER_MODE = "gemini"
            self.history.append(("[SYSTEM]", "Đã chuyển sang chế độ nhận diện qua Gemini Web."))
            self.update_log_display()
        else:
            WHISPER_MODE = "local"
            # Trích xuất tên model từ Local (small) -> small
            model_name = selected.split("(")[1].split(")")[0]
            if LOCAL_MODEL_NAME != model_name or model is None:
                LOCAL_MODEL_NAME = model_name
                model = None
                # Tải/khởi chạy model bất đồng bộ ngay lập tức khi thay đổi chế độ/model
                threading.Thread(target=load_local_model, args=(model_name,), daemon=True).start()
            else:
                self.history.append(("[SYSTEM]", f"Đã chọn model cục bộ ({LOCAL_MODEL_NAME})."))
                self.update_log_display()

    def update_gemini_status_ui(self, status_msg: str):
        if status_msg.startswith("CONNECTED:"):
            profile = status_msg.split("CONNECTED:", 1)[1]
            self.set_status_state("idle", f"Gemini: {profile}")
        elif status_msg.startswith("CONNECTING:"):
            profile = status_msg.split("CONNECTING:", 1)[1]
            self.status_text_label.setText(f"Đang kết nối {profile}...")
        elif status_msg.startswith("FAILED_LOCK:"):
            profile = status_msg.split("FAILED_LOCK:", 1)[1]
            self.set_status_state("error", f"Gemini: {profile} (Bị khóa)")
        elif status_msg.startswith("FAILED_LOGIN:"):
            profile = status_msg.split("FAILED_LOGIN:", 1)[1]
            self.set_status_state("error", f"Gemini: {profile} (Chưa login)")
        elif status_msg.startswith("FAILED:"):
            profile = status_msg.split("FAILED:", 1)[1]
            self.set_status_state("error", f"Gemini: {profile} (Lỗi kết nối)")
        else: # FAILED or DISCONNECTED
            self.set_status_state("error", "Lỗi Gemini")

    def check_server_connection_async(self):
        if WHISPER_MODE != "online" or not API_URL:
            return
            
        self.server_refresh_btn.setEnabled(False)
        self.server_status_label.setText("Checking...")
        self.server_status_label.setStyleSheet("color: #ffa500; font-size: 11px; font-weight: bold; font-family: 'Segoe UI', sans-serif; background: transparent; border: none;")
        self.server_status_led.setStyleSheet("background-color: #ffa500; border-radius: 4px;")
        
        def run_check():
            import requests
            try:
                r = requests.get(API_URL.rsplit('/', 1)[0], timeout=3)
                is_ok = r.status_code < 500
            except Exception:
                is_ok = False
                
            if is_ok:
                subtitle_queue.put("[SYSTEM]\nSERVER_STATUS:OK")
            else:
                subtitle_queue.put("[SYSTEM]\nSERVER_STATUS:ERROR")
                
        threading.Thread(target=run_check, daemon=True).start()

    def update_server_status_ui(self, status):
        if not hasattr(self, "server_status_led"):
            return
        self.server_refresh_btn.setEnabled(True)
        if status == "OK":
            self.server_status_led.setStyleSheet("background-color: #2ecc71; border-radius: 4px;")
            self.server_status_label.setText("Server OK")
            self.server_status_label.setStyleSheet("color: #2ecc71; font-size: 11px; font-weight: bold; font-family: 'Segoe UI', sans-serif; background: transparent; border: none;")
        else:
            self.server_status_led.setStyleSheet("background-color: #e74c3c; border-radius: 4px;")
            self.server_status_label.setText("Server Lỗi")
            self.server_status_label.setStyleSheet("color: #e74c3c; font-size: 11px; font-weight: bold; font-family: 'Segoe UI', sans-serif; background: transparent; border: none;")

    def update_server_status_visibility(self):
        if hasattr(self, "server_status_widget"):
            if WHISPER_MODE == "online":
                self.server_status_widget.show()
                self.check_server_connection_async()
            else:
                self.server_status_widget.hide()

    def toggle_context_panel(self):
        if hasattr(self, "context_panel"):
            is_visible = self.context_panel.isVisible()
            self.context_panel.setVisible(not is_visible)
            self.apply_theme_and_font()

    def analyze_meeting_context(self):
        desc = self.context_desc_input.toPlainText().strip()
        if not desc:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng nhập mô tả ngữ cảnh cuộc họp trước!")
            return
            
        self.analyze_context_btn.setEnabled(False)
        self.analyze_context_btn.setText("⏳ Đang phân tích bằng AI...")
        self.context_prompt_output.setPlainText("AI đang phân tích và thiết lập cấu trúc prompt. Vui lòng đợi...")
        
        def run_analysis():
            prompt = (
                "Bạn là một chuyên gia Prompt Engineering cao cấp. Hãy phân tích đoạn mô tả cuộc họp dưới đây của người dùng:\n"
                f"\"{desc}\"\n\n"
                "Hãy thiết lập và viết một cấu trúc System Instruction chi tiết, chuẩn hóa bằng tiếng Anh (hoặc song ngữ Anh-Việt) "
                "dành cho AI dịch thuật cuộc họp thời gian thực. Cấu trúc phải tuân thủ định dạng XML chặt chẽ với các thẻ cụ thể như sau:\n"
                "<system_instructions>\n"
                "  <identity>\n"
                "    <!-- Vai trò, tính cách và thương hiệu của AI dịch thuật -->\n"
                "  </identity>\n\n"
                "  <product_context>\n"
                "    <!-- Ngữ cảnh về sản phẩm, phần mềm, mục tiêu dự án được thảo luận -->\n"
                "  </product_context>\n\n"
                "  <meeting_context>\n"
                "    <!-- Bối cảnh cuộc họp: tiến trình, các bên thảo luận, mốc thời gian như Go Live -->\n"
                "  </meeting_context>\n\n"
                "  <terminology_rules>\n"
                "    <!-- Quy tắc dịch các thuật ngữ chuyên môn quan trọng được nhắc tới -->\n"
                "  </terminology_rules>\n"
                "</system_instructions>\n\n"
                "Hãy trả về CHỈ duy nhất cấu trúc XML trên, không viết thêm bất kỳ lời giải thích, mở bài, kết bài hay ký tự nào ngoài XML. Sử dụng ví dụ cụ thể của người dùng để điền chi tiết chính xác."
            )
            
            result_prompt = None
            
            if DEEPSEEK_API_KEY:
                try:
                    import requests
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
                    }
                    payload = {
                        "model": "deepseek-chat",
                        "messages": [
                            {"role": "system", "content": "You are a professional prompt engineer. Generate the requested XML prompt block."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3
                    }
                    r = requests.post("https://api.deepseek.com/chat/completions", headers=headers, json=payload, timeout=15)
                    if r.status_code == 200:
                        result_prompt = r.json()["choices"][0]["message"]["content"].strip()
                except Exception as e:
                    print(f"[Context AI] DeepSeek failed: {e}")
                    
            if not result_prompt and GEMINI_API_KEY:
                try:
                    import requests
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
                    payload = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.2}
                    }
                    r = requests.post(url, json=payload, timeout=15)
                    if r.status_code == 200:
                        result_prompt = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                except Exception as e:
                    print(f"[Context AI] Gemini API failed: {e}")
                    
            if not result_prompt and gemini_chat:
                try:
                    result_prompt = translate_with_gemini(prompt)
                except Exception as e:
                    print(f"[Context AI] Gemini Web failed: {e}")
                    
            if not result_prompt:
                result_prompt = (
                    "<system_instructions>\n"
                    "  <identity>\n"
                    "    You are an intelligent real-time translator assistant.\n"
                    "  </identity>\n"
                    "  <product_context>\n"
                    f"    Meeting Topic: {desc}\n"
                    "  </product_context>\n"
                    "</system_instructions>"
                )
                
            if result_prompt.startswith("```xml"):
                result_prompt = result_prompt.split("```xml", 1)[1]
            if result_prompt.startswith("```"):
                result_prompt = result_prompt.split("```", 1)[1]
            if result_prompt.endswith("```"):
                result_prompt = result_prompt.rsplit("```", 1)[0]
            result_prompt = result_prompt.strip()
            
            subtitle_queue.put(f"[SYSTEM]\nCONTEXT_GENERATED:{result_prompt}")
            
        threading.Thread(target=run_analysis, daemon=True).start()

    def apply_meeting_context(self):
        prompt_content = self.context_prompt_output.toPlainText().strip()
        desc_content = self.context_desc_input.toPlainText().strip()
        global CURRENT_CONTEXT_PROMPT
        if not prompt_content:
            CURRENT_CONTEXT_PROMPT = ""
            save_settings({
                "meeting_context_prompt": "",
                "meeting_context_desc": ""
            })
            QMessageBox.information(self, "Thông báo", "Đã xóa bỏ ngữ cảnh cuộc họp hiện tại.")
        else:
            CURRENT_CONTEXT_PROMPT = prompt_content
            save_settings({
                "meeting_context_prompt": CURRENT_CONTEXT_PROMPT,
                "meeting_context_desc": desc_content
            })
            QMessageBox.information(self, "Thành công", "Đã áp dụng ngữ cảnh cuộc họp vào luồng dịch AI!")

    def get_flag_pixmap(self, lang_code):
        code = lang_code.upper()
        if code == "EN":
            filename = "US.png"
        elif code == "JA":
            filename = "JP.png"
        elif code == "VI":
            filename = "VN.png"
        else:
            return None
            
        flag_path = os.path.join(BASE_DIR, "flag-icon", filename)
        if os.path.exists(flag_path):
            pixmap = QPixmap(flag_path)
            return pixmap.scaled(18, 12, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        return None

    def get_flag_html(self, lang_code):
        from PyQt6.QtCore import QUrl
        code = lang_code.upper()
        if code == "EN":
            filename = "US.png"
        elif code == "JA":
            filename = "JP.png"
        elif code == "VI":
            filename = "VN.png"
        else:
            return f"[{lang_code}]"
            
        flag_path = os.path.join(BASE_DIR, "flag-icon", filename)
        if os.path.exists(flag_path):
            file_url = QUrl.fromLocalFile(flag_path).toString()
            return f'<img src="{file_url}" width="16" height="12" style="vertical-align: middle; margin-right: 4px;"/>'
        return f"[{lang_code}]"

    def update_lang_status_ui(self):
        src_code = LANGUAGES.get(SOURCE_LANG, {}).get("google", "auto").split("-")[0].upper()
        if src_code == "AUTO":
            src_code = "Auto"
        tgt_code = LANGUAGES.get(TARGET_LANG, {}).get("google", "vi").split("-")[0].upper()
        
        if hasattr(self, "src_lang_badge"):
            pix = self.get_flag_pixmap(src_code)
            if pix:
                self.src_lang_badge.setPixmap(pix)
            else:
                self.src_lang_badge.setText(src_code)
        if hasattr(self, "tgt_lang_badge"):
            pix = self.get_flag_pixmap(tgt_code)
            if pix:
                self.tgt_lang_badge.setPixmap(pix)
            else:
                self.tgt_lang_badge.setText(tgt_code)

    def swap_languages(self):
        global SOURCE_LANG, TARGET_LANG
        if "Tự động" in SOURCE_LANG:
            src = "Tiếng Anh (English)"
        else:
            src = SOURCE_LANG
            
        tgt = TARGET_LANG
        
        # Tráo đổi
        SOURCE_LANG = tgt
        TARGET_LANG = src
        
        save_settings({
            "source_lang": SOURCE_LANG,
            "target_lang": TARGET_LANG
        })
        self.update_lang_status_ui()
        self.update_log_display()

    def on_settings_file_changed(self, path):
        if path not in self.watcher.files():
            self.watcher.addPath(path)
        QTimer.singleShot(100, self.reload_settings_and_ui)

    def reload_settings_and_ui(self):
        try:
            global WHISPER_MODE, LOCAL_MODEL_NAME, model
            settings = load_settings()
            update_translation_config()
            self.rebuild_mode_select()
            self.update_lang_status_ui()
            self.update_server_status_visibility()
            
            # Nếu dùng local và model chưa load hoặc cấu hình đã thay đổi model khác
            if WHISPER_MODE == "local":
                if model is None or CURRENT_LOADED_MODEL_NAME != LOCAL_MODEL_NAME:
                    model = None
                    threading.Thread(target=load_local_model, args=(LOCAL_MODEL_NAME,), daemon=True).start()
                    
            self.apply_theme_and_font()
            self.update_log_display()
            
            print(f"[Watcher] Đã cập nhật cấu hình ({SOURCE_LANG} ➔ {TARGET_LANG})")
        except Exception as e:
            print(f"[Watcher Error] Không thể tải lại cấu hình: {e}")

    def stop_listening_by_system(self, reason: str):
        global IS_LISTENING
        IS_LISTENING = False
        self.set_status_state("error", "Lỗi: " + reason)
        self.apply_theme_and_font()
        self.history.append(("[SYSTEM]", f"Đã dừng ghi âm do: {reason}"))
        self.update_log_display()
        QMessageBox.critical(self, "Lỗi Server", f"Đã dừng ghi âm.\nLý do: {reason}")

    def update_subtitle(self):
        new_items = []
        try:
            while True:
                txt = subtitle_queue.get_nowait()
                new_items.append(txt)
        except queue.Empty:
            pass

        if new_items:
            has_subtitles = False
            for item in new_items:
                parts = item.split('\n', 1)
                en_text = parts[0]
                vi_text = parts[1] if len(parts) > 1 else ""
                
                if en_text == "[SYSTEM]":
                    if vi_text.startswith("STOP_LISTENING:"):
                        reason = vi_text.split("STOP_LISTENING:", 1)[1]
                        self.stop_listening_by_system(reason)
                    elif vi_text.startswith("SERVER_STATUS:"):
                        status = vi_text.split("SERVER_STATUS:", 1)[1]
                        self.update_server_status_ui(status)
                    elif vi_text.startswith("SERVER_ERROR:"):
                        reason = vi_text.split("SERVER_ERROR:", 1)[1]
                        self.update_server_status_ui("ERROR")
                        self.history.append(("[SYSTEM]", f"Mất kết nối Server Colab: {reason} (Đang tự động thử lại, hoặc nhấn ↻ để kiểm tra kết nối lại)"))
                        has_subtitles = True
                    elif vi_text.startswith("CONTEXT_GENERATED:"):
                        generated_prompt = vi_text.split("CONTEXT_GENERATED:", 1)[1]
                        self.analyze_context_btn.setEnabled(True)
                        self.analyze_context_btn.setText("Phân tích Ngữ cảnh")
                        self.context_prompt_output.setPlainText(generated_prompt)
                    else:
                        is_gemini = False
                        for prefix in ["CONNECTED:", "CONNECTING:", "FAILED_LOCK:", "FAILED_LOGIN:", "FAILED:"]:
                            if vi_text.startswith(prefix):
                                is_gemini = True
                                break
                        if is_gemini:
                            self.update_gemini_status_ui(vi_text)
                        else:
                            self.history.append(("[SYSTEM]", vi_text))
                            has_subtitles = True
                            if "Đã tải và kích hoạt thành công model" in vi_text:
                                QMessageBox.information(self, "Tải Model Hoàn Tất", vi_text)
                elif en_text == "[INTERIM]":
                    draft_parts = vi_text.split('\n', 1)
                    src_draft = draft_parts[0]
                    tgt_draft = draft_parts[1] if len(draft_parts) > 1 else ""
                    self.interim_subtitle = (src_draft, tgt_draft)
                    has_subtitles = True
                else:
                    self.interim_subtitle = None
                    self.history.append((en_text, vi_text))
                    has_subtitles = True
            
            if has_subtitles:
                # Giới hạn lịch sử lưu tối đa 50 câu
                if len(self.history) > 50:
                    self.history = self.history[-50:]
                self.update_log_display()

    def update_log_display(self):
        mode = self.mode_select.currentText()
        if not mode:
            return
            
        scrollbar = self.log_view.verticalScrollBar()
        old_value = scrollbar.value()
        is_at_bottom = (scrollbar.maximum() - scrollbar.value()) <= 50 if scrollbar.maximum() > 0 else True
            
        # Nếu chọn không dịch, cưỡng ép hiển thị chỉ ngôn ngữ gốc
        is_none_translation = (LANGUAGES.get(TARGET_LANG, {}).get("google") == "none")
        if is_none_translation:
            src_clean = get_clean_lang_name(SOURCE_LANG)
            if "1 dòng" in mode:
                mode = f"Chỉ {src_clean} 1 dòng"
            else:
                mode = f"Chỉ {src_clean}"
            
        src_name = LANGUAGES.get(SOURCE_LANG, {}).get("name_vi", "tiếng Anh").capitalize()
        tgt_name = LANGUAGES.get(TARGET_LANG, {}).get("name_vi", "tiếng Việt").capitalize()
        
        src_code = LANGUAGES.get(SOURCE_LANG, {}).get("google", "auto").split("-")[0].upper()
        if src_code == "AUTO":
            src_code = "Auto"
        tgt_code = LANGUAGES.get(TARGET_LANG, {}).get("google", "vi").split("-")[0].upper()
        
        html_content = ""
        is_one_line = "1 dòng" in mode
        
        # Determine colors based on active theme
        if THEME_MODE == "light":
            latest_src_color = "#1F1F1F" # Charcoal
            latest_tgt_color = "#0B57D0" # Royal Blue
            old_src_color = "#5F6368"
            old_tgt_color = "rgba(11, 87, 208, 0.6)"
            draft_en_color = "rgba(0, 0, 0, 0.45)"
            draft_vi_color = "rgba(11, 87, 208, 0.55)"
        else:
            latest_src_color = DRACULA['foreground']
            latest_tgt_color = DRACULA['yellow']
            old_src_color = DRACULA['comment']
            old_tgt_color = DRACULA['comment']
            draft_en_color = "#6C7A9C"
            draft_vi_color = f"rgba({DRACULA['green_rgb']}, 0.55)"
            
        if "Song ngữ" in mode:
            non_system_subs = [(en, vi) for en, vi in self.history if en != "[SYSTEM]"]
            if is_one_line:
                combined_en_parts = [item[0] for item in non_system_subs]
                combined_vi_parts = [item[1] for item in non_system_subs]
                if self.interim_subtitle:
                    en_draft, vi_draft = self.interim_subtitle
                    if en_draft:
                        combined_en_parts.append(f"<i><span style='color: {draft_en_color}; font-family: {FONT_FAMILY_SRC};'>{en_draft}</span></i>")
                    if vi_draft:
                        combined_vi_parts.append(f"<i><span style='color: {draft_vi_color}; font-family: {FONT_FAMILY_TGT};'>{vi_draft}</span></i>")
                
                combined_en = " ".join(combined_en_parts)
                combined_vi = " ".join(combined_vi_parts)
                en_style = f"color: {latest_src_color}; font-size: {FONT_SIZE_SRC}px; font-weight: bold; font-family: {FONT_FAMILY_SRC};"
                vi_style = f"color: {latest_tgt_color}; font-size: {FONT_SIZE_TGT}px; font-weight: bold; font-family: {FONT_FAMILY_TGT};"
                html_content = f"""
                <div style="padding: 10px; min-height: 50px;">
                    {self.get_flag_html(src_code)} 
                    <span style="{en_style}">{combined_en}</span><br/>
                    {self.get_flag_html(tgt_code)} 
                    <span style="{vi_style}">{combined_vi}</span>
                </div>
                """
            else:
                for idx, (en, vi) in enumerate(self.history):
                    is_latest = (idx == len(self.history) - 1)
                    if en == "[SYSTEM]":
                        html_content += f"""
                        <div style="margin-bottom: 8px; text-align: center;">
                            <span style="color: {DRACULA['orange']}; font-size: 13px; font-style: italic; font-weight: bold;">⚠️ {vi}</span>
                        </div>
                        """
                    else:
                        en_color = latest_src_color if is_latest else old_src_color
                        vi_color = latest_tgt_color if is_latest else old_tgt_color
                        en_style = f"color: {en_color}; font-size: {FONT_SIZE_SRC}px; font-weight: {'bold' if is_latest else 'normal'}; font-family: {FONT_FAMILY_SRC};"
                        vi_style = f"color: {vi_color}; font-size: {FONT_SIZE_TGT}px; font-weight: bold; font-family: {FONT_FAMILY_TGT};"
                        
                        html_content += f"""
                        <div style="margin-bottom: 8px;">
                            {self.get_flag_html(src_code)} 
                            <span style="{en_style}">{en}</span><br/>
                            {self.get_flag_html(tgt_code)} 
                            <span style="{vi_style}">{vi}</span>
                        </div>
                        """
                    if idx < len(self.history) - 1:
                        html_content += f'<hr style="border: 0; border-top: 1px solid rgba(98, 114, 164, 30); margin: 6px 0;"/>'
                
                # Append interim draft at the very bottom
                if self.interim_subtitle:
                    en_draft, vi_draft = self.interim_subtitle
                    if html_content:
                        html_content += f'<hr style="border: 0; border-top: 1px dashed rgba(98, 114, 164, 20); margin: 6px 0;"/>'
                    en_style = f"color: {draft_en_color}; font-style: italic; font-size: {FONT_SIZE_SRC}px; font-family: {FONT_FAMILY_SRC};"
                    vi_style = f"color: {draft_vi_color}; font-style: italic; font-size: {FONT_SIZE_TGT}px; font-family: {FONT_FAMILY_TGT};"
                    html_content += f"""
                    <div style="margin-bottom: 8px;">
                        {self.get_flag_html(src_code)} 
                        <span style="{en_style}">{en_draft}</span><br/>
                        {self.get_flag_html(tgt_code)} 
                        <span style="{vi_style}">{vi_draft}</span>
                    </div>
                    """
                        
        elif src_name in mode:  # Chỉ Ngôn ngữ gốc (Source)
            non_system_subs = [en for en, vi in self.history if en != "[SYSTEM]"]
            if is_one_line:
                if self.interim_subtitle:
                    en_draft, _ = self.interim_subtitle
                    if en_draft:
                        non_system_subs.append(f"<i><span style='color: {draft_en_color}; font-family: {FONT_FAMILY_SRC};'>{en_draft}</span></i>")
                if non_system_subs:
                    combined_en = " ".join(non_system_subs)
                    en_style = f"color: {latest_src_color}; font-size: {FONT_SIZE_SRC}px; font-weight: bold; font-family: {FONT_FAMILY_SRC};"
                    html_content = f"""
                    <div style="padding: 10px; min-height: 50px;">
                        <span style="{en_style}">{combined_en}</span>
                    </div>
                    """
            else:
                for idx, (en, vi) in enumerate(self.history):
                    is_latest = (idx == len(self.history) - 1)
                    if en == "[SYSTEM]":
                        html_content += f"""
                        <div style="margin-bottom: 8px; text-align: center;">
                            <span style="color: {DRACULA['orange']}; font-size: 13px; font-style: italic; font-weight: bold;">⚠️ {vi}</span>
                        </div>
                        """
                    else:
                        en_color = latest_src_color if is_latest else old_src_color
                        en_style = f"color: {en_color}; font-size: {FONT_SIZE_SRC}px; font-weight: {'bold' if is_latest else 'normal'}; font-family: {FONT_FAMILY_SRC};"
                        html_content += f"""
                        <div style="margin-bottom: 8px;">
                            {self.get_flag_html(src_code)} 
                            <span style="{en_style}">{en}</span>
                        </div>
                        """
                    if idx < len(self.history) - 1:
                        html_content += f'<hr style="border: 0; border-top: 1px solid rgba(98, 114, 164, 30); margin: 6px 0;"/>'
                
                # Append interim draft
                if self.interim_subtitle:
                    en_draft, _ = self.interim_subtitle
                    if html_content:
                        html_content += f'<hr style="border: 0; border-top: 1px dashed rgba(98, 114, 164, 20); margin: 6px 0;"/>'
                    en_style = f"color: {draft_en_color}; font-style: italic; font-size: {FONT_SIZE_SRC}px; font-family: {FONT_FAMILY_SRC};"
                    html_content += f"""
                    <div style="margin-bottom: 8px;">
                        {self.get_flag_html(src_code)} 
                        <span style="{en_style}">{en_draft}</span>
                    </div>
                    """
                        
        else:  # Chỉ Ngôn ngữ dịch (Target)
            non_system_subs = [vi for en, vi in self.history if en != "[SYSTEM]"]
            if is_one_line:
                if self.interim_subtitle:
                    _, vi_draft = self.interim_subtitle
                    if vi_draft:
                        non_system_subs.append(f"<i><span style='color: {draft_vi_color}; font-family: {FONT_FAMILY_TGT};'>{vi_draft}</span></i>")
                if non_system_subs:
                    combined_vi = " ".join(non_system_subs)
                    vi_style = f"color: {latest_tgt_color}; font-size: {FONT_SIZE_TGT}px; font-weight: bold; font-family: {FONT_FAMILY_TGT};"
                    html_content = f"""
                    <div style="padding: 10px; min-height: 50px;">
                        <span style="{vi_style}">{combined_vi}</span>
                    </div>
                    """
            else:
                for idx, (en, vi) in enumerate(self.history):
                    is_latest = (idx == len(self.history) - 1)
                    if en == "[SYSTEM]":
                        html_content += f"""
                        <div style="margin-bottom: 8px; text-align: center;">
                            <span style="color: {DRACULA['orange']}; font-size: 13px; font-style: italic; font-weight: bold;">⚠️ {vi}</span>
                        </div>
                        """
                    else:
                        vi_color = latest_tgt_color if is_latest else old_tgt_color
                        vi_style = f"color: {vi_color}; font-size: {FONT_SIZE_TGT}px; font-weight: bold; font-family: {FONT_FAMILY_TGT};"
                        html_content += f"""
                        <div style="margin-bottom: 8px;">
                            {self.get_flag_html(tgt_code)} 
                            <span style="{vi_style}">{vi}</span>
                        </div>
                        """
                    if idx < len(self.history) - 1:
                        html_content += f'<hr style="border: 0; border-top: 1px solid rgba(98, 114, 164, 30); margin: 6px 0;"/>'
                
                # Append interim draft
                if self.interim_subtitle:
                    _, vi_draft = self.interim_subtitle
                    if html_content:
                        html_content += f'<hr style="border: 0; border-top: 1px dashed rgba(98, 114, 164, 20); margin: 6px 0;"/>'
                    vi_style = f"color: {draft_vi_color}; font-style: italic; font-size: {FONT_SIZE_TGT}px; font-family: {FONT_FAMILY_TGT};"
                    html_content += f"""
                    <div style="margin-bottom: 8px;">
                        {self.get_flag_html(tgt_code)} 
                        <span style="{vi_style}">{vi_draft}</span>
                    </div>
                    """
                        
        if not html_content:
            if IS_LISTENING:
                self.log_view.setHtml(f"<span style='color: {DRACULA['comment']}; font-style: italic;'>Đang nghe âm thanh... (Nói vào micro hoặc bật âm thanh loa)</span>")
            else:
                self.log_view.setHtml(f"<span style='color: {DRACULA['comment']}; font-style: italic;'>Click vào \"Bắt đầu nghe\" để bắt đầu ghi âm và dịch</span>")
        else:
            self.log_view.setHtml(html_content)
            
        # Cuộn xuống cuối chỉ khi người dùng đang ở cuối trang, ngược lại khóa cứng vị trí cũ để chống giật hình
        if is_at_bottom:
            QTimer.singleShot(10, lambda: scrollbar.setValue(scrollbar.maximum()))
        else:
            scrollbar.setValue(old_value)
            QTimer.singleShot(5, lambda: scrollbar.setValue(old_value))

    def browse_download_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu tệp tải về", self.dl_dir_input.text())
        if dir_path:
            self.dl_dir_input.setText(dir_path)

    def browse_cookie_file(self):
        # Create a popup menu to choose cookie source
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {DRACULA['background']};
                color: {DRACULA['foreground']};
                border: 1px solid {DRACULA['comment']};
                border-radius: 6px;
                padding: 4px 0px;
            }}
            QMenu::item {{
                padding: 6px 20px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 12px;
            }}
            QMenu::item:selected {{
                background-color: {DRACULA['current_line']};
            }}
        """)
        
        # Action 1: Browse File
        file_action = QAction("Chọn file Cookie (.txt)...", self)
        def choose_file():
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Chọn file Cookie",
                "",
                "Text Files (*.txt);;All Files (*)"
            )
            if file_path:
                self.dl_cookie_input.setText(file_path)
        file_action.triggered.connect(choose_file)
        menu.addAction(file_action)
        
        menu.addSeparator()
        
        # Browser actions
        browsers = [
            ("Sử dụng Cookie từ Chrome", "chrome"),
            ("Sử dụng Cookie từ Edge", "edge"),
            ("Sử dụng Cookie từ Firefox", "firefox"),
            ("Sử dụng Cookie từ Brave", "brave")
        ]
        
        for name, value in browsers:
            action = QAction(name, self)
            action.triggered.connect(lambda checked, val=value: self.dl_cookie_input.setText(val))
            menu.addAction(action)
            
        menu.addSeparator()
        
        # Clear action
        clear_action = QAction("Xóa Cookie", self)
        clear_action.triggered.connect(lambda: self.dl_cookie_input.clear())
        menu.addAction(clear_action)
        
        # Show menu at global position of the button
        menu.exec(self.dl_cookie_btn.mapToGlobal(QPoint(0, self.dl_cookie_btn.height())))

    def start_download(self):
        raw_text = self.dl_url_input.toPlainText().strip()
        if not raw_text:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng nhập đường dẫn URL!")
            return

        urls = [line.strip() for line in raw_text.split('\n') if line.strip()]
        if not urls:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng nhập đường dẫn URL hợp lệ!")
            return

        save_dir = self.dl_dir_input.text().strip()
        if not save_dir:
            save_dir = str(BASE_DIR / "downloads")
            self.dl_dir_input.setText(save_dir)

        self.download_queue = urls
        self.download_success_count = 0
        self.download_fail_count = 0

        self.dl_log_view.show()
        self.dl_log_view.clear()
        self.dl_log_view.append(f"Danh sách tải ({len(urls)} link):\n" + "\n".join(urls) + "\n\nThư mục lưu: " + save_dir + "\n" + "-" * 50 + "\n")
        
        self.dl_progress_bar.show()
        self.dl_info_label.show()
        
        self.dl_start_btn.setEnabled(False)
        self.dl_cancel_btn.setEnabled(True)
        self.dl_translate_btn.hide()
        
        self.is_downloading = True
        
        cookie_val = self.dl_cookie_input.text().strip()
        save_settings({"dl_cookie_val": cookie_val})
        global DL_COOKIE_VAL
        DL_COOKIE_VAL = cookie_val

        self.start_next_queued_download()

    def start_next_queued_download(self):
        if not self.download_queue:
            self.finalize_batch_download()
            return

        url = self.download_queue.pop(0)
        save_dir = self.dl_dir_input.text().strip()
        format_opt = "audio" if "âm thanh" in self.dl_format_select.currentText().lower() else "video"

        self.dl_log_view.append(f"\n[SYSTEM] Đang tải link: {url}")
        self.dl_progress_bar.setValue(0)
        self.dl_info_label.setText("Đang chuẩn bị...")
        
        self.download_progress = 0.0
        remaining = len(self.download_queue)
        suffix = f" ({remaining} link còn lại)" if remaining > 0 else ""
        self.floating_btn.setToolTip(f"Đang tải: 0.0%{suffix}")
        self.floating_btn.update()

        # Create and start thread
        self.download_thread = YtDlpDownloadThread(
            url=url,
            save_dir=save_dir,
            format_opt=format_opt,
            ffmpeg_path=FFMPEG_PATH,
            yt_dlp_path=YT_DLP_PATH,
            cookie_val=DL_COOKIE_VAL
        )
        self.download_thread.progress.connect(self.on_download_progress)
        self.download_thread.log.connect(self.on_download_log)
        self.download_thread.finished.connect(self.on_download_finished)
        self.download_thread.start()

    def cancel_download(self):
        if self.download_thread and self.download_thread.isRunning():
            self.dl_log_view.append("\n[SYSTEM] Đang yêu cầu hủy tải xuống và dừng toàn bộ danh sách...")
            self.download_queue.clear()
            self.download_thread.cancel()
            self.dl_cancel_btn.setEnabled(False)

    def on_download_log(self, text):
        self.dl_log_view.append(text)
        # Scroll to bottom
        self.dl_log_view.verticalScrollBar().setValue(
            self.dl_log_view.verticalScrollBar().maximum()
        )

    def on_download_progress(self, percent, status_txt):
        self.download_progress = percent
        self.dl_progress_bar.setValue(int(percent))
        self.dl_info_label.setText(status_txt)
        
        # Update floating button progress
        remaining = len(self.download_queue)
        suffix = f" | {remaining} link còn lại" if remaining > 0 else ""
        self.floating_btn.setToolTip(f"Đang tải: {percent:.1f}% ({status_txt}){suffix}")
        self.floating_btn.update()

    def on_download_finished(self, success, result_str):
        if success:
            self.download_success_count += 1
            self.dl_progress_bar.setValue(100)
            self.dl_info_label.setText("Tải thành công!")
            self.dl_log_view.append(f"\n[SYSTEM] TẢI THÀNH CÔNG: {result_str}")
            
            # Save the downloaded file path for quick translate button
            self.last_downloaded_file = result_str
            self.dl_translate_btn.show()
        else:
            self.download_fail_count += 1
            self.dl_info_label.setText("Tải thất bại")
            if result_str == "Đã hủy tải.":
                self.dl_log_view.append("\n[SYSTEM] ĐÃ HỦY TẢI XUỐNG.")
            else:
                self.dl_log_view.append(f"\n[SYSTEM] TẢI THẤT BẠI: {result_str}")
            
        if self.download_queue:
            QTimer.singleShot(1000, self.start_next_queued_download)
        else:
            self.finalize_batch_download()

    def finalize_batch_download(self):
        self.is_downloading = False
        self.download_progress = 0.0
        self.floating_btn.setToolTip("HT Tool Pro")
        self.floating_btn.update()
        
        self.dl_start_btn.setEnabled(True)
        self.dl_cancel_btn.setEnabled(False)
        
        total_downloads = self.download_success_count + self.download_fail_count
        if total_downloads > 0:
            msg = f"Hoàn tất tải danh sách URL!\n- Thành công: {self.download_success_count}\n- Thất bại: {self.download_fail_count}"
            self.dl_log_view.append(f"\n[SYSTEM] {msg.replace('\n', ' ')}")
            self.dl_info_label.setText(f"Xong (Thành công: {self.download_success_count})")
            QMessageBox.information(self, "Tải hoàn tất", msg)
        else:
            self.dl_info_label.setText("Đã hủy")

    def quick_translate_downloaded_file(self):
        if hasattr(self, "last_downloaded_file") and self.last_downloaded_file and os.path.exists(self.last_downloaded_file):
            # Switch to Tab 2 (Dịch từ File)
            self.tabs.setCurrentIndex(1)
            # Load the file into the file translation tab
            self.on_files_selected([self.last_downloaded_file], append=False)


# ===============================
# BACKGROUND DOWNLOAD WORKER
# ===============================

class YtDlpDownloadThread(QThread):
    progress = pyqtSignal(float, str)
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, url, save_dir, format_opt, ffmpeg_path, yt_dlp_path, cookie_val=""):
        super().__init__()
        self.url = url
        self.save_dir = save_dir
        self.format_opt = format_opt # "audio" or "video"
        self.ffmpeg_path = ffmpeg_path
        self.yt_dlp_path = yt_dlp_path
        self.cookie_val = cookie_val
        self.process = None
        self.is_cancelled = False

    def run(self):
        try:
            # Create save directory if not exist
            os.makedirs(self.save_dir, exist_ok=True)

            # Check for FFMPEG and FFPROBE
            ffmpeg_loc = ""
            if self.ffmpeg_path and os.path.exists(self.ffmpeg_path):
                ffmpeg_loc = self.ffmpeg_path
            else:
                local_ffmpeg = BASE_DIR / "ffmpeg.exe"
                if local_ffmpeg.exists():
                    ffmpeg_loc = str(local_ffmpeg)

            ffprobe_loc = ""
            local_ffprobe = BASE_DIR / "ffprobe.exe"
            if local_ffprobe.exists():
                ffprobe_loc = str(local_ffprobe)

            # Try to load ffmpeg from imageio-ffmpeg package if local_ffmpeg not found
            if not ffmpeg_loc:
                self.log.emit("[FFMPEG] Đang tìm kiếm ffmpeg từ thư viện imageio-ffmpeg...")
                try:
                    import imageio_ffmpeg
                    pkg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                    if pkg_exe and os.path.exists(pkg_exe):
                        self.log.emit("[FFMPEG] Tìm thấy ffmpeg từ imageio-ffmpeg. Đang sao chép vào thư mục ứng dụng...")
                        local_ffmpeg = BASE_DIR / "ffmpeg.exe"
                        import shutil
                        shutil.copy(pkg_exe, str(local_ffmpeg))
                        ffmpeg_loc = str(local_ffmpeg)
                except Exception as e:
                    self.log.emit(f"[FFMPEG Warning] Không thể lấy từ imageio-ffmpeg: {e}")

            # Try to auto-download missing ffmpeg & ffprobe from ffbinaries.com
            if not ffmpeg_loc or not ffprobe_loc:
                self.log.emit("[FFMPEG] Kiểm tra: Thiếu ffmpeg.exe hoặc ffprobe.exe. Bắt đầu tự động tải từ ffbinaries.com...")
                import zipfile
                import io
                import requests
                
                urls = {
                    "ffmpeg.exe": [
                        "https://github.com/ffbinaries/ffbinaries-prebuilt/releases/download/v4.4.1/ffmpeg-4.4.1-win-64.zip",
                        "https://node.ffbinaries.com/bin/windows-64/ffmpeg.zip"
                    ],
                    "ffprobe.exe": [
                        "https://github.com/ffbinaries/ffbinaries-prebuilt/releases/download/v4.4.1/ffprobe-4.4.1-win-64.zip",
                        "https://node.ffbinaries.com/bin/windows-64/ffprobe.zip"
                    ]
                }
                
                for filename, url_list in urls.items():
                    dest_path = BASE_DIR / filename
                    if dest_path.exists():
                        if filename == "ffmpeg.exe":
                            ffmpeg_loc = str(dest_path)
                        else:
                            ffprobe_loc = str(dest_path)
                        continue
                    
                    self.log.emit(f"[FFMPEG] Đang tải {filename}...")
                    download_success = False
                    for url in url_list:
                        try:
                            r = requests.get(url, stream=True, timeout=30)
                            r.raise_for_status()
                            
                            z_data = io.BytesIO(r.content)
                            with zipfile.ZipFile(z_data) as z:
                                for name in z.namelist():
                                    if name.lower() == filename.lower() or name.lower().endswith(filename.lower()):
                                        with open(dest_path, "wb") as f_out:
                                            f_out.write(z.read(name))
                            self.log.emit(f"[FFMPEG] Tải thành công và giải nén {filename}!")
                            if filename == "ffmpeg.exe":
                                ffmpeg_loc = str(dest_path)
                            else:
                                ffprobe_loc = str(dest_path)
                            download_success = True
                            break
                        except Exception as ex:
                            self.log.emit(f"[FFMPEG Cảnh báo] Thử tải {filename} từ {url} thất bại: {str(ex)}")
                    
                    if not download_success:
                        self.log.emit(f"[FFMPEG Lỗi] Tất cả liên kết tải {filename} đều thất bại.")

            # Determine if FFMPEG is available
            ffmpeg_available = bool(ffmpeg_loc) and bool(ffprobe_loc)
            if ffmpeg_available:
                self.log.emit(f"[FFMPEG] Đã kích hoạt FFMPEG tại: {ffmpeg_loc} và FFPROBE tại: {ffprobe_loc}")
            else:
                self.log.emit("[FFMPEG Cảnh báo] Thiếu FFMPEG hoặc FFPROBE. Hệ thống sẽ tự động tải định dạng gốc trực tiếp (m4a đối với âm thanh, mp4 đối với video) để tránh lỗi postprocessing.")

            # Build command
            cmd = []
            
            # 1. yt-dlp path
            if self.yt_dlp_path and os.path.exists(self.yt_dlp_path):
                cmd.append(self.yt_dlp_path)
            else:
                # Check next to execution script
                local_exe = BASE_DIR / "yt-dlp.exe"
                if local_exe.exists():
                    cmd.append(str(local_exe))
                else:
                    # Check system path
                    import shutil
                    sys_exe = shutil.which("yt-dlp")
                    if sys_exe:
                        cmd.append(sys_exe)
                    else:
                        # Fallback to python module
                        cmd.extend([sys.executable, "-m", "yt_dlp"])

            # 2. Add URL
            cmd.append(self.url)

            # 3. Add output path template
            out_tmpl = os.path.join(self.save_dir, "%(title)s.%(ext)s")
            cmd.extend(["-o", out_tmpl])

            # 4. ffmpeg path
            if ffmpeg_available:
                cmd.extend(["--ffmpeg-location", ffmpeg_loc])

            # 5. Format options
            if self.format_opt == "audio":
                if ffmpeg_available:
                    cmd.extend(["-x", "--audio-format", "mp3", "--audio-quality", "0"])
                else:
                    cmd.extend(["-f", "ba[ext=m4a]/ba"])
            else:
                if ffmpeg_available:
                    cmd.extend(["-f", "bv*+ba/b", "--merge-output-format", "mp4"])
                else:
                    cmd.extend(["-f", "b[ext=mp4]/b"])

            # 5.5 Add Cookie Options if present
            if self.cookie_val:
                if self.cookie_val.lower() in ["chrome", "firefox", "edge", "opera", "safari", "vivaldi", "brave"]:
                    cmd.extend(["--cookies-from-browser", self.cookie_val.lower()])
                else:
                    cmd.extend(["--cookies", self.cookie_val])

            # 6. Extra options for clean parsing & update interval
            cmd.extend(["--newline", "--progress", "--no-colors"])

            log_msg = f"Khởi chạy lệnh: {' '.join(cmd)}\n"
            self.log.emit(log_msg)

            # Start subprocess
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                startupinfo=startupinfo,
                cwd=self.save_dir
            )

            # Patterns to extract details
            progress_re = re.compile(r"\[download\]\s+(\d+(?:\.\d+)?)%(?:\s+of\s+(\S+))?(?:\s+at\s+(\S+))?(?:\s+ETA\s+(\S+))?")
            dest_re = re.compile(r"\[download\]\s+Destination:\s+(.+)")
            merger_re = re.compile(r"\[(?:Merger|ExtractAudio)\]\s+(?:Merging formats into|Destination)\s+\"?(.*?)\"?$")
            
            downloaded_file = ""

            while True:
                if self.is_cancelled:
                    break
                line = self.process.stdout.readline()
                if not line:
                    break
                
                line_str = line.strip()
                if not line_str:
                    continue

                self.log.emit(line_str)

                # Parse destination file
                m_dest = dest_re.search(line_str)
                if m_dest:
                    downloaded_file = m_dest.group(1).strip()
                
                m_merger = merger_re.search(line_str)
                if m_merger:
                    downloaded_file = m_merger.group(1).strip()

                # Parse progress
                m_prog = progress_re.search(line_str)
                if m_prog:
                    percent = float(m_prog.group(1))
                    total_size = m_prog.group(2) or "Unknown"
                    speed = m_prog.group(3) or "Unknown"
                    eta = m_prog.group(4) or "Unknown"
                    
                    status_txt = f"{total_size} | {speed} | ETA: {eta}"
                    self.progress.emit(percent, status_txt)

            self.process.wait()
            ret_code = self.process.returncode

            # Windows WinError 32 File Locking Recovery Logic
            if ret_code != 0 and not self.is_cancelled:
                import glob
                temp_files = glob.glob(os.path.join(self.save_dir, "*.temp.*"))
                for temp_file in temp_files:
                    parts = temp_file.rsplit(".temp.", 1)
                    if len(parts) == 2:
                        final_file = parts[0] + "." + parts[1]
                        if not os.path.exists(final_file) or os.path.getsize(final_file) == 0:
                            self.log.emit(f"[Sửa lỗi WinError 32] Phát hiện file tạm chưa được đổi tên do Windows lock file: {os.path.basename(temp_file)}")
                            self.log.emit(f"[Sửa lỗi WinError 32] Đang tự động thử đổi tên lại sang: {os.path.basename(final_file)}...")
                            rename_ok = False
                            for attempt in range(15):  # Try for up to 7.5 seconds
                                try:
                                    if os.path.exists(final_file):
                                        os.remove(final_file)
                                    os.rename(temp_file, final_file)
                                    self.log.emit("[Sửa lỗi WinError 32] Đổi tên thành công! Tiếp tục tiến trình.")
                                    rename_ok = True
                                    downloaded_file = final_file
                                    ret_code = 0  # Override failure!
                                    break
                                except Exception:
                                    time.sleep(0.5)
                            if not rename_ok:
                                self.log.emit("[Sửa lỗi WinError 32] Thử đổi tên thất bại. Vui lòng tắt các cửa sổ Folder chứa file hoặc kiểm tra phần mềm diệt virus.")

            if self.is_cancelled:
                self.log.emit("Đã hủy tải xuống bởi người dùng.")
                self.finished.emit(False, "Đã hủy tải.")
                return

            if ret_code == 0:
                if downloaded_file:
                    for suffix in [".f137", ".f251", ".f18", ".temp", ".part"]:
                        if downloaded_file.endswith(suffix):
                            downloaded_file = downloaded_file[:-len(suffix)]
                
                # If we don't have downloaded_file, find the newest file in self.save_dir
                if not downloaded_file or not os.path.exists(downloaded_file):
                    import glob
                    files = glob.glob(os.path.join(self.save_dir, "*"))
                    if files:
                        downloaded_file = max(files, key=os.path.getmtime)
                
                self.finished.emit(True, downloaded_file)
            else:
                self.finished.emit(False, f"Lỗi tiến trình tải (Exit code: {ret_code})")

        except Exception as e:
            self.log.emit(f"Lỗi hệ thống: {str(e)}")
            self.finished.emit(False, str(e))

    def cancel(self):
        self.is_cancelled = True
        if self.process:
            try:
                self.process.terminate()
                self.process.kill()
            except Exception as e:
                print(f"[Cancel Error] {e}")


# ===============================
# MAIN
# ===============================

def main():

    threading.Thread(
        target=speech_worker,
        daemon=True
    ).start()

    p = pyaudio.PyAudio()
    stream = None

    if RECORD_MODE == "loopback":
        try:
            # 1. Lấy tên thiết bị phát mặc định thực tế của hệ thống (qua MME)
            try:
                mme_default = p.get_default_output_device_info()
                mme_default_name = mme_default["name"]
                clean_default_name = mme_default_name.split("(")[0].strip()
            except Exception as mme_err:
                print(f"Không thể lấy thiết bị MME: {mme_err}")
                clean_default_name = None

            # 2. Lấy thiết bị phát mặc định WASAPI
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        except Exception as e:
            print(f"Không thể lấy thiết bị phát mặc định từ WASAPI: {e}")
            p.terminate()
            sys.exit(1)

        # 3. Tìm thiết bị loopback tương ứng
        loopback_dev = None
        
        # Thử tìm loopback khớp với thiết bị phát thực tế của Windows đang chọn
        if clean_default_name:
            mme_compare = mme_default_name[:30].lower().strip()
            for loopback in p.get_loopback_device_info_generator():
                loopback_compare = loopback["name"].lower()
                if mme_compare in loopback_compare:
                    loopback_dev = loopback
                    break

        # Nếu không tìm thấy, fallback về mặc định của WASAPI
        if not loopback_dev:
            if default_speakers["isLoopbackDevice"]:
                loopback_dev = default_speakers
            else:
                for loopback in p.get_loopback_device_info_generator():
                    if default_speakers["name"] in loopback["name"]:
                        loopback_dev = loopback
                        break

        if not loopback_dev:
            print("Không tìm thấy thiết bị Loopback của loa mặc định.")
            p.terminate()
            sys.exit(1)

        print(f"Đang ghi âm âm thanh hệ thống từ loa: {loopback_dev['name']}")

        native_rate = int(loopback_dev["defaultSampleRate"])
        channels = loopback_dev["maxInputChannels"]

        def pyaudio_callback(in_data, frame_count, time_info, status):
            try:
                # Nếu không ở trạng thái lắng nghe, bỏ qua dữ liệu âm thanh
                if not IS_LISTENING:
                    return (None, pyaudio.paContinue)
                
                audio_data = np.frombuffer(in_data, dtype=np.float32)
                
                # Trích xuất sang mono nếu stereo
                if channels > 1:
                    audio_data = audio_data.reshape(-1, channels)
                    audio_data = np.mean(audio_data, axis=1)
                
                # Resample về SAMPLE_RATE Hz
                resampled_audio = resample(audio_data, native_rate, SAMPLE_RATE)
                audio_queue.put(resampled_audio)
            except Exception as e:
                print(f"Lỗi trong loopback callback: {e}", file=sys.stderr)
            return (None, pyaudio.paContinue)

        stream = p.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=native_rate,
            input=True,
            input_device_index=loopback_dev["index"],
            stream_callback=pyaudio_callback
        )

    elif RECORD_MODE == "mic":
        try:
            default_input = p.get_default_input_device_info()
            print(f"Đang ghi âm từ Microphone mặc định: {default_input['name']}")

            channels = 1
            native_rate = SAMPLE_RATE # PyAudio tự động resample cho ngõ vào mic chuẩn

            def pyaudio_callback(in_data, frame_count, time_info, status):
                try:
                    # Nếu không ở trạng thái lắng nghe, bỏ qua dữ liệu âm thanh
                    if not IS_LISTENING:
                        return (None, pyaudio.paContinue)
                    
                    audio_data = np.frombuffer(in_data, dtype=np.float32)
                    audio_queue.put(audio_data)
                except Exception as e:
                    print(f"Lỗi trong mic callback: {e}", file=sys.stderr)
                return (None, pyaudio.paContinue)

            stream = p.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=native_rate,
                input=True,
                input_device_index=default_input["index"],
                stream_callback=pyaudio_callback
            )
        except Exception as e:
            print(f"Không thể mở Microphone mặc định: {e}")
            p.terminate()
            sys.exit(1)

    stream.start_stream()

    app = QApplication(sys.argv)
    app.setStyleSheet(f"""
        QToolTip {{
            color: {DRACULA['foreground']};
            background-color: {DRACULA['background']};
            border: 1px solid {DRACULA['comment']};
            border-radius: 4px;
            padding: 4px;
            font-family: 'Segoe UI', sans-serif;
            font-size: 11px;
        }}
    """)

    # Try to find a valid icon from multiple sources
    icon_paths = [
        RESOURCE_DIR / "icon.png",
        BASE_DIR / "icon.png",
        RESOURCE_DIR / "icon.ico",
        BASE_DIR / "icon.ico"
    ]
    icon = None
    for path in icon_paths:
        if path.exists():
            test_icon = QIcon(str(path))
            if not test_icon.isNull():
                icon = test_icon
                break
    if icon:
        app.setWindowIcon(icon)

    overlay = Overlay()
    overlay.show()

    try:
        sys.exit(app.exec())
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except:
            pass
        p.terminate()

# ===============================

if __name__ == "__main__":
    main()