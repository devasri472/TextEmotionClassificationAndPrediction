import streamlit as st
import altair as alt
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from track_utils import create_page_visited_table, add_page_visited_details, view_all_page_visited_details, add_prediction_details, view_all_prediction_details, create_emotionclf_table, IST  # Import IST from track_utils

# Load Model
pipe_en = joblib.load(open("./Model/voting_classifier_model.pkl", "rb"))

# Mapping integers to emotion labels (updated: removed 'happy')
label_to_emotion = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "joy",
    4: "neutral",
    5: "sad",
    6: "shame",
    7: "surprise"  # Added 'surprise'
}

# Emoji dictionary for each emotion
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱",
    "joy": "😂", "neutral": "😐",
    "sad": "😔", "shame": "😳", "surprise": "😮"  # Confirmed 'surprise'
}

# Function to predict emotion
def predict_emotions(docx):
    results = pipe_en.predict([docx])
    return label_to_emotion.get(results[0], "unknown")

def get_prediction_proba(docx):
    results = pipe_en.predict_proba([docx])
    return results

def main():
    st.title("Emotion Classifier App")
    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Initialize database tables
    create_page_visited_table()
    create_emotionclf_table()

    if choice == "Home":
        add_page_visited_details("Home", datetime.now(IST))
        st.subheader("Emotion Detection in Text")

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)

            # Predict emotion and probability
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            add_prediction_details(raw_text, prediction, np.max(probability), datetime.now(IST))

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict.get(prediction, "")
                st.write(f"{prediction}: {emoji_icon}")
                st.write(f"Confidence: {np.max(probability):.2f}")

            with col2:
                st.success("Prediction Probability")

                proba_df = pd.DataFrame(probability, columns=pipe_en.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                # Map emotion labels to display names
                proba_df_clean['emotions'] = proba_df_clean['emotions'].map(label_to_emotion)

                fig = alt.Chart(proba_df_clean).mark_bar().encode(
                    x=alt.X('emotions:N', title='Emotion'),
                    y=alt.Y('probability:Q', title='Probability'),
                    color=alt.Color('emotions:N', legend=None)  # Color each bar based on emotion
                ).properties(
                   width=600,
                   height=400
                ).configure_axis(
                   labelFontSize=12,
                   titleFontSize=14
                ).configure_view(
                   strokeWidth=0
                )

                st.altair_chart(fig, use_container_width=True)

    elif choice == "Monitor":
        add_page_visited_details("Monitor", datetime.now(IST))
        st.subheader("Monitor App")

        with st.expander("Page Metrics"):
            page_visited_details = pd.DataFrame(view_all_page_visited_details(), columns=['Page Name', 'Time of Visit'])
            st.dataframe(page_visited_details)

            pg_count = page_visited_details['Page Name'].value_counts().rename_axis('Page Name').reset_index(name='Counts')
            c = alt.Chart(pg_count).mark_bar().encode(x='Page Name', y='Counts', color='Page Name')
            st.altair_chart(c, use_container_width=True)

            p = px.pie(pg_count, values='Counts', names='Page Name')
            st.plotly_chart(p, use_container_width=True)

        with st.expander('Emotion Classifier Metrics'):
            df_emotions = pd.DataFrame(view_all_prediction_details(), columns=['Rawtext', 'Prediction', 'Probability', 'Time_of_Visit'])
            st.dataframe(df_emotions)

            prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
            pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction', y='Counts', color='Prediction')
            st.altair_chart(pc, use_container_width=True)

    else:
        add_page_visited_details("About", datetime.now(IST))

        st.write("Welcome to the Emotion Detection in Text App! This application utilizes the power of natural language processing and machine learning to analyze and identify emotions in textual data.")

        st.subheader("Our Mission")
        st.write("At Emotion Detection in Text, our mission is to provide a user-friendly and efficient tool that helps individuals and organizations understand the emotional content hidden within text...")

        st.subheader("How It Works")
        st.write("When you input text into the app, our system processes it and applies advanced natural language processing algorithms to extract meaningful features from the text...")

        st.subheader("Key Features:")
        st.markdown("##### 1. Real-time Emotion Detection\n##### 2. Confidence Score\n##### 3. User-friendly Interface")
        st.subheader("Applications")
        st.markdown("The Emotion Detection in Text App has a wide range of applications across various industries and domains...")

if __name__ == '__main__':
    main()
