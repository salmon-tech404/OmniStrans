import asyncio
from fastapi import FastAPI, WebSocket
import uvicorn
import numpy as np
from faster_whisper import WhisperModel
from pyngrok import ngrok
import nest_asyncio
import json
import os

# Tự động cài đặt speechbrain nếu thiếu
try:
    import speechbrain
except ImportError:
    import subprocess
    import sys
    print("[SERVER] Đang tự động cài đặt speechbrain...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "speechbrain"])
    import speechbrain

# Khởi tạo FastAPI
app = FastAPI()

# 1. BỔ SUNG CỔNG TIẾP ĐÓN GỐC (Sửa lỗi 404 Not Found khi ping)
@app.get("/")
def root():
    return {"status": "ok", "message": "WebSocket STT Server is running!"}

# Đặt ngrok authtoken từ Cell 2
try:
    if 'NGROK_AUTH_TOKEN' in globals() and NGROK_AUTH_TOKEN:
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        print("[SERVER] Đã nhận thành công NGROK_AUTH_TOKEN từ Cell 2!")
    else:
        print("[WARNING] Không tìm thấy NGROK_AUTH_TOKEN. Hãy chắc chắn bạn đã chạy Cell 2 trước!")
except Exception as token_err:
    print(f"[WARNING] Lỗi thiết lập ngrok token: {token_err}")

# Tải mô hình Whisper trên GPU của Colab
print("Loading Whisper Model...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print("Whisper Model Loaded Successfully!")

# Khởi tạo mô hình nhận diện giọng nói SpeechBrain (ECAPA-TDNN) TRÊN CPU để tránh xung đột luồng CUDA với Whisper
print("Loading Speaker Recognition Model (SpeechBrain) on CPU...")
try:
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    speaker_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        run_opts={"device": "cpu"}
    )
    print("Speaker Recognition Model Loaded Successfully!")
except Exception as e:
    speaker_model = None
    print(f"[WARNING] Không thể tải mô hình SpeechBrain: {e}")

import torch.nn.functional as F

# 4. HÀM PHÂN TÍCH VÀ SO KHỚP VÉC-TƠ GIỌNG NÓI (COSINE SIMILARITY) TRÊN CPU
def identify_speaker(audio_np, known_speakers, threshold=0.70):
    if speaker_model is None:
        return "Speaker 1"
        
    try:
        # Nếu đoạn âm thanh quá ngắn (dưới 1.2 giây), thông tin giọng nói không đủ để phân tích chính xác.
        # Ta gán luôn cho người nói gần nhất để tránh nhận diện nhầm thành người mới (người nói ảo).
        if len(audio_np) < 16000 * 1.2:
            if known_speakers:
                return known_speakers[-1]["id"]
            return "Speaker 1"

        # Chuyển đổi và chạy trên CPU để đảm bảo an toàn tuyệt đối và không xung đột CUDA
        signal = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0).to("cpu")
        # Trích xuất embedding đặc trưng giọng nói
        with torch.no_grad():
            embeddings = speaker_model.encode_batch(signal)
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            embeddings = embeddings.squeeze().cpu() # Véc-tơ 1D 192 chiều
            
        best_score = -1.0
        best_speaker_id = None
        
        # So sánh với dấu vân tay giọng nói của những người đã nói trước đó
        for spk in known_speakers:
            score = torch.dot(embeddings, spk["embedding"]).item()
            if score > best_score:
                best_score = score
                best_speaker_id = spk["id"]
                
        # Nếu độ tương đồng cao hơn ngưỡng (0.70), xác nhận là người cũ
        if best_score >= threshold:
            return best_speaker_id
        else:
            # Nếu không giống ai, đăng ký là người nói mới
            new_id = f"Speaker {len(known_speakers) + 1}"
            known_speakers.append({"id": new_id, "embedding": embeddings})
            print(f"[SERVER] Đăng ký người nói mới: {new_id} (Độ tương đồng lớn nhất: {best_score:.4f})")
            return new_id
    except Exception as ex:
        print(f"[SERVER] Lỗi trích xuất dấu vân tay giọng nói: {ex}")
        return "Speaker 1"


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[SERVER] Client connected via WebSocket!")
    
    # KHỞI TẠO BỘ ĐỆM TÍCH LŨY, KHÓA GPU MUTEX VÀ DANH SÁCH SPEAKER
    audio_buffer = bytearray()
    last_interim_bytes = 0
    transcribe_lock = asyncio.Lock()
    known_speakers = [] # Bộ lưu trữ giọng nói cho phiên kết nối hiện tại
    
    try:
        # Đọc tham số ngôn ngữ gửi từ app Client
        first_msg = await websocket.receive_text()
        params = json.loads(first_msg)
        whisper_lang = params.get("language", None)
        print(f"[SERVER] Client target language: {whisper_lang}")

        while True:
            message = await websocket.receive()
            if "bytes" in message:
                data = message["bytes"]
                audio_buffer.extend(data)
                
                # Chỉ chạy dịch nháp khi tích lũy thêm 0.5s dữ liệu mới và GPU đang rảnh rỗi
                if len(audio_buffer) - last_interim_bytes >= 16000:
                    if not transcribe_lock.locked():
                        async with transcribe_lock:
                            last_interim_bytes = len(audio_buffer)
                            audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                            
                            segments, info = await asyncio.to_thread(
                                model.transcribe,
                                audio_np, 
                                language=whisper_lang, 
                                beam_size=1,
                                vad_filter=True
                            )
                            text = "".join(seg.text.strip() + " " for seg in segments).strip()
                            if text:
                                await websocket.send_json({"type": "interim", "text": text})
                        
            elif "text" in message:
                text_data = message["text"]
                cmd = json.loads(text_data)
                if cmd.get("type") == "finalize":
                    # Chốt câu chất lượng cao nhất (beam_size=5) cho TOÀN BỘ câu nói
                    async with transcribe_lock:
                        if audio_buffer:
                            audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                            
                            # 1. Chép lời câu nói
                            segments, info = await asyncio.to_thread(
                                model.transcribe,
                                audio_np, 
                                language=whisper_lang, 
                                beam_size=5,
                                vad_filter=True
                            )
                            text = "".join(seg.text.strip() + " " for seg in segments).strip()
                            
                            # 2. Phân biệt người nói (Speaker Diarization)
                            speaker_id = "Speaker 1"
                            if text:
                                speaker_id = await asyncio.to_thread(
                                    identify_speaker,
                                    audio_np,
                                    known_speakers
                                )
                        else:
                            text = ""
                            speaker_id = "Speaker 1"
                        
                        await websocket.send_json({
                            "type": "final", 
                            "speaker": speaker_id, 
                            "text": text
                        })
                        
                        # Reset hoàn toàn bộ đệm để chuẩn bị cho câu tiếp theo
                        audio_buffer = bytearray()
                        last_interim_bytes = 0
                
    except Exception as e:
        print(f"[SERVER] Connection closed or error: {e}")
    finally:
        print("[SERVER] Client disconnected!")

# Khởi chạy server và lấy link ngrok
nest_asyncio.apply()
public_url = ngrok.connect(8000, bind_tls=True, domain="sedative-apron-wooing.ngrok-free.dev")
ws_url = public_url.public_url.replace("https://", "wss://").replace("http://", "ws://")
print("\n" + "="*60)
print(f" ĐỊA CHỈ API WEBSOCKET CỦA BẠN (Dán vào ô API_URL trong Cài đặt app):")
print(f" {ws_url}/ws")
print("="*60 + "\n")

# Chạy uvicorn và cấu hình VÔ HIỆU HÓA HOÀN TOÀN Ping-Pong Keepalive từ cả hai phía
config = uvicorn.Config(
    app=app, 
    host="0.0.0.0", 
    port=8000, 
    ws_ping_interval=None, 
    ws_ping_timeout=None
)
server = uvicorn.Server(config)
await server.serve()
