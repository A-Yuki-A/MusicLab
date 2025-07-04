import streamlit as st
try:
    from pydub import AudioSegment
except ModuleNotFoundError:
    st.error("pydub モジュールがインストールされていません。requirements.txt に 'pydub' を追加してください。")
    st.stop()
import numpy as np
import matplotlib.pyplot as plt
import librosa
import tempfile
import soundfile as sf

# ── ffmpeg/ffprobe のパス指定 ──
AudioSegment.converter = "/usr/bin/ffmpeg"
AudioSegment.ffprobe   = "/usr/bin/ffprobe"

# ── ページ設定 ──
st.set_page_config(
    page_title="WaveForge",
    layout="centered"
)

# ── 音声ロード関数 ──
def load_mp3(uploaded_file):
    """
    MP3を一時ファイル経由で読み込み、正規化したNumPy配列とサンプリングレートを返す
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    audio = AudioSegment.from_file(tmp_path, format="mp3")
    sr = audio.frame_rate
    data = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.channels == 2:
        data = data.reshape((-1, 2)).mean(axis=1)
    data /= np.abs(data).max()
    return data, sr

# ── アプリ本体 ──
st.title("WaveForge")

# ファイルアップロード
uploaded_file = st.file_uploader("MP3ファイルをアップロード", type="mp3")
if not uploaded_file:
    st.info("MP3ファイルをアップロードしてください。")
    st.stop()

# 音声読み込み
data, orig_sr = load_mp3(uploaded_file)
duration = len(data) / orig_sr

# ── 設定変更 ──
st.write("### 設定変更")
st.markdown("**標本化周波数と量子化ビット数を変えて、音の違いを聴き比べしなさい。**")

# 標本化周波数 ラベル＋説明
st.markdown(
    "<span style='font-weight:bold; color:orange;'>標本化周波数 (Hz)：</span>1秒間に何回の標本点として音の大きさを取り込むかを示します。高いほど細かい音を再現できます。通常音源の標本化周波数は44,100Hz",
    unsafe_allow_html=True
)
# 標本化周波数の下限を4000Hzに設定
target_sr = st.slider("", 4000, 48000, orig_sr if orig_sr >= 4000 else 44100, step=1000)

# 量子化ビット数 ラベル＋説明
st.markdown(
    "<span style='font-weight:bold; color:orange;'>量子化ビット数：</span>各標本点の電圧を何段階に分けて記録するかを示します。ビット数が多いほど音の強弱を滑らかに表現できます。通常音源の量子化ビット数は16bit",
    unsafe_allow_html=True
)
bit_depth = st.slider("", 3, 24, 16, step=1)

# ── 再サンプリングと量子化 ──
rs_data = librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
max_int = 2**(bit_depth - 1) - 1
quantized = np.round(rs_data * max_int) / max_int

# ── 波形比較 ──
st.write("### 波形比較")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)

t_orig = np.linspace(0, duration, num=len(data))
ax1.plot(t_orig, data)
ax1.set(title="Original Waveform", xlabel="Time (s)", ylabel="Amplitude")
ax1.set(xlim=(0, duration), ylim=(-1, 1))

proc_len = min(len(quantized), int(duration * target_sr))
t_proc = np.linspace(0, duration, num=proc_len)
ax2.plot(t_proc, quantized[:proc_len])
ax2.set(title=f"Processed Waveform ({target_sr:,} Hz, {bit_depth:,} ビット)", xlabel="Time (s)", ylabel="Amplitude")
ax2.set(xlim=(0, duration), ylim=(-1, 1))

st.pyplot(fig)

# ── オーディオ再生 ──
st.write("### 再生")
subtype_map = {8: 'PCM_U8', 16: 'PCM_16', 24: 'PCM_24'}
selected_subtype = subtype_map.get(bit_depth, 'PCM_16')

if np.all(quantized == 0):
    st.warning(f"{target_sr} Hz にリサンプリングした結果、無音になってしまいました。標本化周波数を少し上げてください。")
else:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
        sf.write(out.name, quantized, target_sr, subtype=selected_subtype)
        st.audio(out.name, format="audio/wav")

# ── データ量計算 ──
st.write("### データ量計算")
st.markdown("**音のデータ量 = 標本化周波数 (Hz) × 量子化ビット数 (bit) × 時間 (s) × チャンネル数 (ch)**")
st.markdown("**設定を変更したファイルのデータ量＝**")

bytes_size = target_sr * bit_depth * 2 * duration / 8
kb_size = bytes_size / 1024
mb_size = kb_size / 1024

example = f"""
<div style='line-height:1.2;'>
{target_sr:,} Hz × {bit_depth:,} bit × 2 ch × {duration:.2f} 秒 ÷ 8 = {int(bytes_size):,} バイト<br>
{int(bytes_size):,} バイト ÷ 1024 = {kb_size:,.2f} KB<br>
{kb_size:,.2f} KB ÷ 1024 = {mb_size:,.2f} MB
</div>
"""
st.markdown(example, unsafe_allow_html=True)

channel_desc = """
<div style='line-height:1.5; margin-top:20px;'>
- ステレオ(2ch): 左右2つの音声信号を同時に再生します。音に広がりがあります。<br>
- モノラル(1ch): 1つの音声信号で再生します。音の定位は中央になります。
</div>
"""
st.markdown(channel_desc, unsafe_allow_html=True)
