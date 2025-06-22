import streamlit as st
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import io
import tempfile

# ── ここで ffmpeg/ffprobe の場所を指定 ──
# Streamlit Cloud 上では通常以下のパスにインストールされています
AudioSegment.converter = "/usr/bin/ffmpeg"
AudioSegment.ffprobe   = "/usr/bin/ffprobe"

def load_mp3(uploaded_file) -> tuple[np.ndarray, int]:
    """
    1) アップロードされたファイルを一時ファイルに書き出し
    2) pydub で mp3 を読み込み
    3) NumPy 配列とサンプリング周波数を返す
    """
    # 一時ファイルを作成
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # pydub で読み込み
    audio = AudioSegment.from_file(tmp_path, format="mp3")
    sr = audio.frame_rate

    # NumPy 配列に変換
    data = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.channels == 2:
        data = data.reshape((-1, 2))

    # 正規化（-1～1 の範囲に）
    if data.dtype != np.float32:
        data /= np.iinfo(data.dtype).max

    return data, sr

# ── Streamlit アプリ本体 ──
st.title("🎧 MP3 音声解析ツール")

uploaded = st.file_uploader("MP3 ファイルをアップロード", type="mp3")
if uploaded:
    # 読み込み
    data, sr = load_mp3(uploaded)
    st.write(f"**サンプリング周波数:** {sr} Hz")

    # 波形表示
    st.write("### 波形（Waveform）")
    fig, ax = plt.subplots()
    t = np.linspace(0, len(data)/sr, num=len(data))
    if data.ndim == 1:
        ax.plot(t, data)
    else:
        ax.plot(t, data[:,0], label="左チャンネル")
        ax.plot(t, data[:,1], label="右チャンネル")
        ax.legend()
    ax.set_xlabel("時間 (秒)")
    ax.set_ylabel("振幅")
    st.pyplot(fig)

    # スペクトログラム表示
    st.write("### スペクトログラム（Spectrogram）")
    mono = data.mean(axis=1) if data.ndim == 2 else data
    S = librosa.stft(mono)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    fig2, ax2 = plt.subplots()
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax2)
    ax2.set_title("周波数スペクトル (dB)")
    st.pyplot(fig2)

    # 再生
    st.write("### 再生")
    st.audio(uploaded, format="audio/mp3")
