from faster_whisper import WhisperModel
import time

# モデルをロード
model = WhisperModel("large-v3-turbo", download_root="assets/fwhisper")

# 生成の開始時刻を記録
start_time = time.time()

# 音源の読み込み
segments, _ = model.transcribe("sample.wav", beam_size=1, language="ja")

for segment in segments:
    # 開始時刻との差分から経過時間を表示
    print(time.time() - start_time)
    # 生成結果を表示
    print(segment.text)
