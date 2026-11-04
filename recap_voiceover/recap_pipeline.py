#!/usr/bin/env python3
"""
recap_pipeline.py — turn a LOCAL video you have rights to into a narrated recap.

Steps (each is a subcommand, run in order or use `all`):
  frames  : capture a JPG every N seconds + timestamps.csv
  script  : draft Thai narration per scene from the frames (Claude vision API)
  voice   : synthesize narration in YOUR cloned voice (ElevenLabs API)
  build   : cut the video to match each narration clip, mux audio, add subtitles

This tool only reads local files. It does not download anything. Use it only
with footage you own or are licensed to use (your own videos, CC / public
domain, or material whose owner has given you permission).

Requires: ffmpeg + ffprobe on PATH, `pip install requests anthropic`
Env vars: ANTHROPIC_API_KEY (script), ELEVENLABS_API_KEY + ELEVENLABS_VOICE_ID (voice)
"""
import argparse
import base64
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def die(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def probe_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ------------------------------------------------------------------ frames
def cmd_frames(a):
    out = Path(a.work); (out / "frames").mkdir(parents=True, exist_ok=True)
    for f in (out / "frames").glob("*.jpg"):
        f.unlink()
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", a.video,
         "-vf", f"fps=1/{a.interval:g},scale=-2:720", "-q:v", "3",
         out / "frames" / "f_%05d.jpg"])
    files = sorted((out / "frames").glob("f_*.jpg"))
    if not files:
        die("no frames produced")
    with open(out / "timestamps.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["no", "file", "seconds"])
        for i, f in enumerate(files):
            w.writerow([i + 1, f.name, f"{i * a.interval:.1f}"])
    print(f"{len(files)} frames -> {out/'frames'}")


# ------------------------------------------------------------------ script
def cmd_script(a):
    try:
        import anthropic
    except ModuleNotFoundError:
        die("pip install anthropic")
    work = Path(a.work)
    rows = list(csv.DictReader(open(work / "timestamps.csv", encoding="utf-8")))
    step = max(1, a.every)  # use every Nth frame as one scene
    scenes = rows[::step]
    client = anthropic.Anthropic()
    out, prev = [], ""
    for i, r in enumerate(scenes):
        img = base64.standard_b64encode(
            (work / "frames" / r["file"]).read_bytes()).decode()
        msg = client.messages.create(
            model=a.model, max_tokens=300,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": "image/jpeg", "data": img}},
                {"type": "text", "text":
                 f"คุณกำลังเล่าสรุปวิดีโอเป็นภาษาไทยแบบเป็นกันเอง นี่คือฉากที่ {i+1}/"
                 f"{len(scenes)} (วินาทีที่ {r['seconds']}). เรื่องย่อ/บริบทจากผู้ใช้: "
                 f"{a.context or '(ไม่มี)'}\nบทที่เล่าไปแล้วล่าสุด: {prev or '(ยังไม่มี)'}\n"
                 "เขียนบทพากย์ 1-2 ประโยคสำหรับฉากนี้ ต่อเนื่องจากบทก่อนหน้า "
                 "ตอบเฉพาะบทพากย์เท่านั้น"}]}])
        text = msg.content[0].text.strip()
        prev = text
        out.append({"start": float(r["seconds"]), "text": text})
        print(f"[{i+1}/{len(scenes)}] {text}")
    json.dump(out, open(work / "script.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"Edit {work/'script.json'} by hand if you like, then run `voice`.")


# ------------------------------------------------------------------ voice
def cmd_voice(a):
    import requests
    key = os.environ.get("ELEVENLABS_API_KEY")
    voice = a.voice_id or os.environ.get("ELEVENLABS_VOICE_ID")
    if not key or not voice:
        die("set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID (your cloned voice)")
    work = Path(a.work); (work / "audio").mkdir(exist_ok=True)
    script = json.load(open(work / "script.json", encoding="utf-8"))
    for i, s in enumerate(script):
        dest = work / "audio" / f"n_{i:04d}.mp3"
        r = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice}",
            headers={"xi-api-key": key},
            json={"text": s["text"], "model_id": a.tts_model}, timeout=120)
        if r.status_code != 200:
            die(f"TTS failed ({r.status_code}): {r.text[:200]}")
        dest.write_bytes(r.content)
        s["audio"] = dest.name
        s["duration"] = probe_duration(dest)
        print(f"[{i+1}/{len(script)}] {dest.name} {s['duration']:.1f}s")
    json.dump(script, open(work / "script.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


# ------------------------------------------------------------------ build
def cmd_build(a):
    work = Path(a.work); seg_dir = work / "segments"; seg_dir.mkdir(exist_ok=True)
    script = json.load(open(work / "script.json", encoding="utf-8"))
    vid_len = probe_duration(Path(a.video))
    segs, srt, t = [], [], 0.0
    for i, s in enumerate(script):
        if "audio" not in s:
            die("run `voice` first")
        dur = s["duration"] + 0.3
        start = min(s["start"], max(0.0, vid_len - dur))
        seg = seg_dir / f"s_{i:04d}.mp4"
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", a.video,
             "-i", work / "audio" / s["audio"],
             "-map", "0:v:0", "-map", "1:a:0",
             "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,"
                    "pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30",
             "-af", "apad", "-t", f"{dur:.3f}",
             "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac",
             "-ar", "44100", "-ac", "2", seg])
        segs.append(seg)
        srt.append(f"{i+1}\n{srt_time(t)} --> {srt_time(t + s['duration'])}\n{s['text']}\n")
        t += dur
    (work / "recap.srt").write_text("\n".join(srt), encoding="utf-8")
    lst = work / "concat.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in segs))
    out = Path(a.output)
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
         "-safe", "0", "-i", lst, "-c", "copy", out])
    print(f"Done: {out}  (subtitles: {work/'recap.srt'})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["frames", "script", "voice", "build", "all"])
    ap.add_argument("--video", help="local video file you have rights to")
    ap.add_argument("--work", default="recap_work")
    ap.add_argument("--output", default="recap.mp4")
    ap.add_argument("--interval", type=float, default=10)
    ap.add_argument("--every", type=int, default=3,
                    help="use every Nth frame as a narration scene")
    ap.add_argument("--context", default="", help="plot/context notes for the narrator")
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--voice-id", default=None)
    ap.add_argument("--tts-model", default="eleven_multilingual_v2",
                    help="check the model supports Thai before use")
    a = ap.parse_args()
    steps = ["frames", "script", "voice", "build"] if a.step == "all" else [a.step]
    if any(s in ("frames", "build") for s in steps):
        if not a.video or not Path(a.video).is_file():
            die("--video must be an existing local file")
    for s in steps:
        globals()[f"cmd_{s}"](a)


if __name__ == "__main__":
    main()
