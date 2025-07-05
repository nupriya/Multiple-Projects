import streamlit as st
import pandas as pd
import re
from textblob import TextBlob
import matplotlib.pyplot as plt

st.set_page_config(page_title="Chat Sentiment Analyzer", layout="wide")

st.title("💬 Chat Sentiment Analyzer")
st.write("Paste your WhatsApp/Slack-style chat and analyze message sentiments.")

# 1. User input
chat_input = st.text_area("📋 Paste Chat Text Here", height=300)

if st.button("Analyze"):
    if not chat_input.strip():
        st.warning("Please paste some chat text before analyzing.")
    else:
        # 2. Extract dialogue lines using regex
        dialogues = re.findall(r"(Human \d): (.+)", chat_input)
        if not dialogues:
            st.error("No valid dialogue lines found. Make sure it’s in 'Human 1: message' format.")
        else:
            df = pd.DataFrame(dialogues, columns=["Speaker", "Message"])

            # 3. Sentiment analysis
            def get_sentiment(text):
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                if polarity > 0.1:
                    return "Positive"
                elif polarity < -0.1:
                    return "Negative"
                else:
                    return "Neutral"

            df["Sentiment"] = df["Message"].apply(get_sentiment)

            # 4. Display results
            st.subheader("📊 Message Sentiments")
            st.dataframe(df, use_container_width=True)

            # 5. Sentiment distribution
            st.subheader("📈 Sentiment Distribution")
            sentiment_counts = df["Sentiment"].value_counts()
            fig, ax = plt.subplots()
            sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'], ax=ax)
            ax.set_title("Sentiment Count")
            ax.set_ylabel("Number of Messages")
            st.pyplot(fig)

            # 6. Option to download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Sentiment CSV", csv, "chat_sentiment.csv", "text/csv")
