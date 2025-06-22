import streamlit as st
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import io

def load_mp3(file) -> tuple[np.ndarray, int]:
    # pydubでMP3を読み込み、NumPy配列とサンプリング周波数を返す
    audio = AudioSegment.from_file(file, format="mp3")
    data = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.channels == 2:
        data = data.reshape((-1, 2))  # ステレオなら左右に分割
    sr = audio.frame_rate
    # 正規化（-1～1の範囲に）
    data /= np.iinfo(data.dtype).max if data.dtype != np.float32 else 1.0
    return data, sr

st.title("🎧 MP3音声解析ツール")

uploaded = st.file_uploader("MP3ファイルをアップロード", type="mp3")
if uploaded:
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
    # librosaでモノラルに変換しSTFT
    mono = data.mean(axis=1) if data.ndim == 2 else data
    S = librosa.stft(mono)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    fig2, ax2 = plt.subplots()
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax2)
    ax2.set_title("周波数スペクトル (dB)")
    st.pyplot(fig2)
    
    # 音声再生
    st.write("### 再生")
    st.audio(uploaded, format="audio/mp3")
