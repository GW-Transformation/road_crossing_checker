import subprocess
import threading
import time
import os
import sys
import platform

class AlarmWarningPlayer:
    """
    Audio Dispatcher for:
      1. Voice warning on NG: "กรุณาหยุด ชี้นิ้วตามทางแยกให้ถูกต้อง"
      2. Short confirmation chime on Step OK (Step 1, 2, 3)
    """
    def __init__(self, audio_path=None, chime_path=None, cooldown_sec=3.5, enabled=True):
        self.enabled = enabled
        self.cooldown_sec = cooldown_sec
        self.last_alarm_time = 0
        self.last_chime_time = 0
        self.lock = threading.Lock()
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.audio_path = audio_path or os.path.join(base_dir, "assets", "warning_alarm.mp3")
        self.chime_path = chime_path or os.path.join(base_dir, "assets", "step_ok_chime.wav")
        self.os_type = platform.system()

    def play_step_ok(self, step_num=1):
        """Plays short, crisp confirmation tone to signal step is OK"""
        if not self.enabled:
            return False
            
        with self.lock:
            now = time.time()
            if now - self.last_chime_time < 0.25:
                return False
            self.last_chime_time = now

        threading.Thread(target=self._play_sound_file, args=(self.chime_path, "Step OK Chime"), daemon=True).start()
        return True

    def play_alarm(self, reason="Safety Standard NG"):
        """Plays voice warning on safety violation"""
        if not self.enabled:
            return False
            
        with self.lock:
            now = time.time()
            if now - self.last_alarm_time < self.cooldown_sec:
                return False
            self.last_alarm_time = now

        threading.Thread(target=self._play_alarm_worker, daemon=True).start()
        return True

    def _play_sound_file(self, file_path, label="Sound"):
        if not os.path.exists(file_path):
            return
        try:
            if self.os_type == "Darwin": # macOS
                subprocess.run(["afplay", file_path], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif self.os_type == "Linux": # Linux / Raspberry Pi
                subprocess.run(["aplay", "-q", file_path], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif self.os_type == "Windows":
                import winsound
                winsound.PlaySound(file_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            pass

    def _play_alarm_worker(self):
        print(f"[ALARM WARNING] 🔊 Triggered voice alert: 'กรุณาหยุด ชี้นิ้วตามทางแยกให้ถูกต้อง'")
        try:
            if self.os_type == "Darwin":
                if os.path.exists(self.audio_path):
                    subprocess.run(["afplay", self.audio_path], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    subprocess.run(["say", "-v", "Kanya", "กรุณาหยุด ชี้นิ้วตามทางแยกให้ถูกต้อง"], check=False)
            elif self.os_type == "Linux":
                if os.path.exists(self.audio_path):
                    res = subprocess.run(["mpg123", "-q", self.audio_path], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if res.returncode != 0:
                        subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", self.audio_path], check=False)
            elif self.os_type == "Windows":
                os.system(f'start /min "" "{self.audio_path}"')
        except Exception as e:
            print(f"[WARN] Alarm audio error: {e}")
