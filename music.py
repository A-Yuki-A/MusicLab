import streamlit as st
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import io
import tempfile

def load_mp3(uploaded_file) -> tuple[np.ndarray, int]:
    # 1) 一時ファイルを作成してMP3データを書き込む
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # 2) pydubで一時ファイルを読み込む
    audio = AudioSegment.from_file(tmp_path, format="mp3")
    sr = audio.frame_rate

    # 3) NumPy配列に変換
    data = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.channels == 2:
        data = data.reshape((-1, 2))
    # 正規化
    data /= np.iinfo(data.dtype).max if data.dtype != np.float32 else 1.0

    return data, sr

# Streamlit アプリ本体は前と同じです
st.title("🎧 MP3音声解析ツール")

uploaded = st.file_uploader("MP3ファイルをアップロード", type="mp3")
if uploaded:
    data, sr = load_mp3(uploaded)
    st.write(f"**サンプリング周波数:** {sr} Hz")
    # 波形・スペクトログラム・再生のコードは先ほどと同様…
