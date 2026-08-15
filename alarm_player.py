import subprocess
import threading
import queue
import time
import os
import platform

class AlarmWarningPlayer:
    """
    Low-latency Audio Engine:
      1. Step 1 & 2 OK: Short confirmation shot chime
      2. Step 3 (Total OK): Upbeat Level-Up celebration chime
      3. Safety Violation NG: Voice alarm "กรุณาหยุด ชี้นิ้วตามทางแยกให้ถูกต้อง"
    """
    def __init__(self, audio_path=None, chime_path=None, levelup_path=None, cooldown_sec=3.0, enabled=True):
        self.enabled = enabled
        self.cooldown_sec = cooldown_sec
        self.last_alarm_time = 0
        self.last_chime_time = 0
        self.os_type = platform.system()
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.audio_path = audio_path or os.path.join(base_dir, "assets", "warning_alarm.mp3")
        self.chime_path = chime_path or os.path.join(base_dir, "assets", "step_ok_chime.wav")
        self.levelup_path = levelup_path or os.path.join(base_dir, "assets", "level_up_chime.wav")
        
        # Dedicated worker thread for instant non-blocking audio dispatch
        self.audio_queue = queue.Queue(maxsize=10)
        self.worker_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.worker_thread.start()

    def play_step_ok(self, step_num=1):
        if not self.enabled:
            return False
            
        now = time.time()
        if now - self.last_chime_time < 0.15:
            return False
        self.last_chime_time = now

        sound_file = self.levelup_path if step_num >= 3 else self.chime_path
        tag = "LEVEL UP (Step 3 Complete)" if step_num >= 3 else f"Step {step_num} OK Chime"
        
        try:
            self.audio_queue.put_nowait(("sound", sound_file, tag))
            return True
        except queue.Full:
            return False

    def play_alarm(self, reason="Safety Standard NG"):
        if not self.enabled:
            return False
            
        now = time.time()
        if now - self.last_alarm_time < self.cooldown_sec:
            return False
        self.last_alarm_time = now

        try:
            self.audio_queue.put_nowait(("alarm", self.audio_path, reason))
            return True
        except queue.Full:
            return False

    def _audio_loop(self):
        while True:
            item = self.audio_queue.get()
            if item is None:
                break
            action, path, desc = item
            
            try:
                if action == "sound":
                    if self.os_type == "Darwin":
                        subprocess.run(["afplay", path], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    elif self.os_type == "Linux":
                        subprocess.run(["aplay", "-q", path], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    elif self.os_type == "Windows":
                        import winsound
                        winsound.PlaySound(path, winsound.SND_FILENAME)
                        
                elif action == "alarm":
                    print(f"[ALARM WARNING] 🔊 Voice Alert: 'กรุณาหยุด ชี้นิ้วตามทางแยกให้ถูกต้อง' ({desc})")
                    if self.os_type == "Darwin":
                        if os.path.exists(path):
                            subprocess.run(["afplay", path], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        else:
                            subprocess.run(["say", "-v", "Kanya", "กรุณาหยุด ชี้นิ้วตามทางแยกให้ถูกต้อง"], check=False)
                    elif self.os_type == "Linux":
                        if os.path.exists(path):
                            res = subprocess.run(["mpg123", "-q", path], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            if res.returncode != 0:
                                subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path], check=False)
                    elif self.os_type == "Windows":
                        os.system(f'start /min "" "{path}"')
            except Exception as e:
                pass
            finally:
                self.audio_queue.task_done()
