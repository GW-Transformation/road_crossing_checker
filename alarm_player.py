import subprocess
import threading
import time
import os
import sys
import platform

class AlarmWarningPlayer:
    """
    Asynchronous voice warning dispatcher for Safety NG alerts:
    Warning voice: "กรุณาหยุด ชี้นิ้วตามทางแยกให้ถูกต้อง"
    """
    def __init__(self, audio_path=None, cooldown_sec=4.0, enabled=True):
        self.enabled = enabled
        self.cooldown_sec = cooldown_sec
        self.last_play_time = 0
        self.lock = threading.Lock()
        
        if audio_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.audio_path = os.path.join(base_dir, "assets", "warning_alarm.mp3")
        else:
            self.audio_path = audio_path
            
        self.os_type = platform.system()

    def play_alarm(self, reason="Safety Standard NG"):
        if not self.enabled:
            return False
            
        with self.lock:
            now = time.time()
            if now - self.last_play_time < self.cooldown_sec:
                # Still within cooldown
                return False
            self.last_play_time = now

        threading.Thread(target=self._play_worker, daemon=True).start()
        return True

    def _play_worker(self):
        print(f"[ALARM WARNING] 🔊 Triggered voice alert: 'กรุณาหยุด ชี้นิ้วตามทางแยกให้ถูกต้อง'")
        try:
            if self.os_type == "Darwin": # macOS
                if os.path.exists(self.audio_path):
                    subprocess.run(["afplay", self.audio_path], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    # Fallback to macOS say
                    subprocess.run(["say", "-v", "Kanya", "กรุณาหยุด ชี้นิ้วตามทางแยกให้ถูกต้อง"], check=False)
            elif self.os_type == "Linux": # Raspberry Pi / Linux
                if os.path.exists(self.audio_path):
                    # Try mpg123, ffplay, or aplay
                    res = subprocess.run(["mpg123", "-q", self.audio_path], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if res.returncode != 0:
                        subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", self.audio_path], check=False)
                else:
                    print("[WARN] Audio file not found for Linux playback")
            elif self.os_type == "Windows":
                # Windows powershell or cmd playback
                import winsound
                # winsound plays wav, or start file
                os.system(f'start /min "" "{self.audio_path}"')
        except Exception as e:
            print(f"[WARN] Audio playback error: {e}")
