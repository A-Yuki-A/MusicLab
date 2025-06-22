# Settings
st.write("### Settings")
target_sr = st.slider("Sampling Rate (Hz)", 8000, 48000, orig_sr, step=1000)
st.caption("🔊 標本化周波数（Sampling Rate）を下げると、高い音が失われて音がこもった感じになります。")

bit_depth = st.slider("Quantization Bits", 8, 24, 16, step=1)
st.caption("🔉 量子化ビット数（Bit Depth）を下げると、音の滑らかさが失われ、ノイズが増えたように聞こえることがあります。")

st.write(f"Original SR: {orig_sr} Hz → Target SR: {target_sr} Hz | Quantization: {bit_depth}-bit")
