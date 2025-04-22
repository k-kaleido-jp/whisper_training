from optimum.intel.openvino import OVModelForSpeechSeq2Seq 
from transformers import AutoProcessor, pipeline
import torch
import torchaudio

import time

# モデルとトークナイザの読み込み
model = OVModelForSpeechSeq2Seq.from_pretrained("./assets/ov_turbo").to('GPU')
model.compile()
processor = AutoProcessor.from_pretrained("./assets/ov_turbo")
# 音源の読み込み、テンソルに変換
waveform, sample_rate = torchaudio.load("sample.wav")

# サンプリング周期のチェック
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

# テンソルを整形
inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")

# 生成の開始時刻を記録
start_time = time.time()

# モデルにテンソルを渡して生成
with torch.no_grad():
    generated_ids = model.generate(inputs["input_features"], language="ja")

# モデルから生成されたテンソルをトークナイザで文字列に変換
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# 開始時刻との差分から経過時間を表示し、生成結果を表示
print(time.time() - start_time)
print(text)
