import os
import tempfile
import requests
import numpy as np
import wave

class AudioTranscriber:
    def __init__(self, languages_dict, load_local_model_fn, transcribe_with_gemini_web_fn):
        self.languages = languages_dict
        self.load_local_model = load_local_model_fn
        self.transcribe_with_gemini_web = transcribe_with_gemini_web_fn

    def save_wav(self, audio_data, filepath, sample_rate=16000):
        """Save float32 1D numpy array as a standard 16-bit mono PCM WAV file."""
        scaled = np.int16(np.clip(audio_data, -1.0, 1.0) * 32767)
        with wave.open(filepath, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2) # 2 bytes = 16-bit PCM
            w.setframerate(sample_rate)
            w.writeframes(scaled.tobytes())

    def transcribe(self, audio_source, mode, src_lang, local_model_name=None, api_url=None, gemini_chat=None, sample_rate=16000):
        """
        Transcribes audio source.
        
        Parameters:
            audio_source: str (file path) or np.ndarray (1D float32 array)
            mode: str ("online" | "gemini" | "local")
            src_lang: str (key in languages dict)
            local_model_name: str (e.g., "tiny", "base", "small")
            api_url: str (online server API URL)
            gemini_chat: object (Gemini Chat object)
            sample_rate: int (default 16000)
            
        Returns:
            tuple: (original_text, segments)
                - original_text: str (transcribed text, or combined segments text)
                - segments: list or None (raw segments returned from local Whisper model if in-memory audio)
        """
        is_file = isinstance(audio_source, str)
        whisper_lang = self.languages.get(src_lang, {}).get("whisper")

        if mode == "online":
            if not api_url:
                raise ValueError("API_URL is required for online mode")
            
            params = {}
            if whisper_lang:
                params["language"] = whisper_lang

            try:
                if is_file:
                    with open(audio_source, "rb") as f:
                        r = requests.post(
                            api_url,
                            params=params,
                            files={"file": (os.path.basename(audio_source), f)},
                            timeout=60
                        )
                    if r.status_code == 200:
                        return (r.json().get("text") or "").strip(), None
                    else:
                        error_text = r.text[:300]
                        if "ngrok" in error_text.lower() or r.status_code == 404:
                            raise Exception("Server online chưa hoạt động (lỗi 404/ngrok offline)")
                        raise Exception(f"Server trả về mã lỗi {r.status_code}")
                else:
                    # Numpy array: save to temp wav first
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp_path = tmp.name
                    try:
                        self.save_wav(audio_source, tmp_path, sample_rate)
                        with open(tmp_path, "rb") as f:
                            r = requests.post(
                                api_url,
                                params=params,
                                files={"file": ("chunk.wav", f)},
                                timeout=30
                            )
                        if r.status_code == 200:
                            return (r.json().get("text") or "").strip(), None
                        else:
                            error_text = r.text[:300]
                            if "ngrok" in error_text.lower() or r.status_code == 404:
                                raise Exception("Server online chưa hoạt động (lỗi 404/ngrok offline)")
                            raise Exception(f"Server trả về mã lỗi {r.status_code}")
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
            except Exception as e:
                # If it's already a clean/custom exception we raised, re-raise it
                if "Server online chưa hoạt động" in str(e) or "Server trả về mã lỗi" in str(e):
                    raise e
                # Otherwise, wrap it in a friendly message
                err_str = str(e)
                if "connection" in err_str.lower() or "ssl" in err_str.lower() or "max retries" in err_str.lower():
                    raise Exception("Không thể kết nối tới server (Server online chưa hoạt động hoặc ngrok offline)")
                raise Exception(f"Không thể kết nối tới server (Lỗi: {e})")

        elif mode == "gemini":
            if not gemini_chat:
                raise Exception("Chưa kết nối Gemini Web. Hãy chọn profile và kết nối.")
            
            if is_file:
                text = self.transcribe_with_gemini_web(audio_source)
                return text, None
            else:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    self.save_wav(audio_source, tmp_path, sample_rate)
                    text = self.transcribe_with_gemini_web(tmp_path)
                    return text, None
                finally:
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass

        else: # local mode
            if not local_model_name:
                raise ValueError("local_model_name is required for local mode")
            active_model = self.load_local_model(local_model_name)
            if active_model is None:
                raise Exception("Không thể tải model Whisper cục bộ.")

            segments, info = active_model.transcribe(
                audio_source,
                language=whisper_lang,
                vad_filter=True
            )
            if is_file:
                segment_list = list(segments)
                original_text = " ".join(seg.text.strip() for seg in segment_list)
                return original_text, None
            else:
                return "", list(segments)
