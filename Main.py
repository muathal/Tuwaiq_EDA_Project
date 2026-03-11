import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import langid
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- SETUP ---
st.set_page_config(page_title="Blog EDA", layout="wide")


@st.cache_resource
def load_nltk():
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')


load_nltk()


# --- PREPROCESSING LOGIC ---
def remove_url(text):
    return re.sub(r'(https?://\S+|www\.\S+|urlink)', '', text)


def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r'\W', ' ', text)


def preprocess_text(text):
    text = remove_url(text)
    text = clean_text(text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in tokens if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(w) for w in filtered])


@st.cache_data
def process_data():
    # Load and Sample
    df = pd.read_csv("data/blogtext.csv")
    ndf = df.drop(columns="sign").sample(frac=0.05, random_state=32, ignore_index=True)
    ndf.drop_duplicates(inplace=True)

    # Language
    ndf["Lang"] = ndf["text"].apply(lambda x: langid.classify(x)[0])
    ndf["Lang"] = ndf["Lang"].apply(lambda x: "en" if x == "en" else "non-en")

    # Text Cleaning
    ndf["text_cleaned"] = ndf["text"].apply(preprocess_text)

    # Date
    ndf["date_cleaned"] = pd.to_datetime(ndf["date"], errors="coerce")
    ndf.dropna(subset=["date_cleaned"], inplace=True)
    ndf["year"] = ndf["date_cleaned"].dt.year
    ndf["month"] = ndf["date_cleaned"].dt.month
    ndf["day"] = ndf["date_cleaned"].dt.day

    # Age Groups
    def ageGroup(x):
        if 12 <= x <= 18:
            return "Teen"
        elif 19 <= x <= 29:
            return "Young Adult"
        elif 30 <= x <= 49:
            return "Adult"
        else:
            return "Old"

    ndf["age group"] = ndf["age"].apply(ageGroup)
    return ndf


# --- APP EXECUTION ---
st.title("Blog Analysis Dashboard")
ndf = process_data()

# Header: Initial checking and cleaning
st.header("1. Initial Checking & Cleaning")
st.write(f"Dataframe Shape: {ndf.shape}")
st.dataframe(ndf.head())

# Header: Language Distribution
st.header("2. Language Analysis")
fig_lang, ax_lang = plt.subplots()
sns.lineplot(data=ndf[ndf["Lang"] == "non-en"].groupby("year").size(), ax=ax_lang)
ax_lang.set_title("Increase in Non-English Users by Year")
st.pyplot(fig_lang)

# Header: Topic count
st.header("3. Topic Analysis")
fig_topic, ax_topic = plt.subplots(figsize=(10, 5))
sns.countplot(data=ndf, y="topic", order=ndf["topic"].value_counts().index, ax=ax_topic)
st.pyplot(fig_topic)

# Header: Age Group
st.header("4. Age Group Analysis")
fig_age, ax_age = plt.subplots()
sns.countplot(data=ndf, x="age group", order=["Teen", "Young Adult", "Adult", "Old"], ax=ax_age)
st.pyplot(fig_age)

# Header: Topic by Age Group
st.header("5. Topic by Age Group")
fig_tag, ax_tag = plt.subplots(figsize=(10, 6))
sns.countplot(data=ndf, y="topic", hue="age group", ax=ax_tag)
st.pyplot(fig_tag)

# Header: Gender Analysis
st.header("6. Gender Analysis")
col1, col2 = st.columns(2)
with col1:
    fig_gen, ax_gen = plt.subplots()
    sns.countplot(data=ndf, x="gender", ax=ax_gen)
    st.pyplot(fig_gen)
with col2:
    fig_gent, ax_gent = plt.subplots()
    sns.countplot(data=ndf, y="topic", hue="gender", ax=ax_gent)
    st.pyplot(fig_gent)

# Header: Top 5 Users
st.header("7. Top 5 Active Users")
st.table(ndf["id"].value_counts().head(5))