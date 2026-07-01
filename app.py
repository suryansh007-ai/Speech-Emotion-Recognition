import streamlit as st
import tempfile
from predict import pred_emotion
st.set_page_config(
    page_title="Speech Emotion Recognition",
    layout='wide'
)
st.title('Speech Emotion Recognition')
st.write("Upload a speech recording and let the model predict the speaker's emotion")
with st.sidebar:
    st.header("About")

    st.write("""
    **Model**
    Random Forest

    **Dataset**
    RAVDESS

    **Features**
    - MFCC
    - Chroma
    - Mel Spectrogram
    """)
uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav"]
)
if uploaded_file is not None:

    st.audio(uploaded_file)
    if st.button("Analyze Emotion"):
        with st.spinner("Analyzing..."):
            with tempfile.NamedTemporaryFile(delete=False,suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path=tmp_file.name
            emotion,confidence=pred_emotion(temp_path)
        st.success(f"Predicted Emotion: {emotion}")
        st.success(f"Confidence: {confidence}")