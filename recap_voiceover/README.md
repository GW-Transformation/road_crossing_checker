# Recap voiceover pipeline

Turns a **local video you have rights to** into a recap narrated in your own (cloned) voice.
It does not download anything.

```bash
pip install requests anthropic
export ANTHROPIC_API_KEY=...        # for `script`
export ELEVENLABS_API_KEY=...       # for `voice`
export ELEVENLABS_VOICE_ID=...      # your cloned voice (record 1-5 min of yourself in ElevenLabs)

python recap_pipeline.py frames --video my_video.mp4 --interval 10
python recap_pipeline.py script --context "เรื่องย่อสั้น ๆ" --every 3
# edit recap_work/script.json if you want
python recap_pipeline.py voice
python recap_pipeline.py build --video my_video.mp4 --output recap.mp4
```

Output: `recap.mp4` plus `recap_work/recap.srt`. Check that the chosen `--tts-model` supports Thai.
Only use footage you own or are licensed to use.
