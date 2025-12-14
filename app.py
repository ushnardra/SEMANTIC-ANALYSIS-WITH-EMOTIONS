import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

pipe_lr = joblib.load(open("model/balanced_emotion.pkl", "rb"))
emojis = {
    "anger": "😠",
    "fear": "😨",
    "joy": "😂",
    "sad": "😔",
    "suprise": "😮",
    "love":"😘"
}

def predict_emotion(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotion in Text")

    with st.form(key='emotion_clf_form'):
        raw_text = st.text_area("Type Your Text Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        prediction = predict_emotion(raw_text)

        st.success("Original Text")
        st.write(raw_text)

        st.success("Prediction")
        emoji_icon = emojis[prediction]
        st.write(f"{prediction} : {emoji_icon}")

if __name__ == '__main__':
    main()