import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- SETUP ---
st.set_page_config(page_title="Blog EDA Dashboard", layout="wide")


# --- DATA LOADING ---
@st.cache_data  # Cache the reading so it's super fast
def load_cleaned_data():
    # Read the file you saved in Step 1
    return pd.read_csv("data/cleaned_blogtext.csv")


# --- APP EXECUTION ---
st.title("📊 Blog Analysis Dashboard (Cleaned Data)")

try:
    ndf = load_cleaned_data()

    # 1. Show Data
    st.header("1. Cleaned Data Preview")
    st.dataframe(ndf.head())

    # 2. Visualizations (Every plot you wanted)
    st.header("2. Exploratory Visualizations")

    tab1, tab2, tab3 = st.tabs(["Demographics", "Topic Trends", "Activity"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Gender Distribution")
            fig, ax = plt.subplots()
            sns.countplot(data=ndf, x="gender", ax=ax)
            st.pyplot(fig)
        with col2:
            st.subheader("Age Group Distribution")
            fig, ax = plt.subplots()
            sns.countplot(data=ndf, x="age group", order=["Teen", "Young Adult", "Adult", "Old"], ax=ax)
            st.pyplot(fig)

    with tab2:
        st.subheader("Popularity by Topic")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=ndf, y="topic", order=ndf["topic"].value_counts().index, ax=ax)
        st.pyplot(fig)

        st.subheader("Topics by Gender")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=ndf, y="topic", hue="gender", ax=ax)
        st.pyplot(fig)

    with tab3:
        st.subheader("Non-English Posts by Year")
        # Assuming you kept the 'year' and 'Lang' columns
        non_en_trend = ndf[ndf["Lang"] == "non-en"].groupby("year").size()
        st.line_chart(non_en_trend)

    # 3. Top Users Result
    st.header("3. Top 5 Most Active Users")
    st.table(ndf["id"].value_counts().head(5))

except FileNotFoundError:
    st.error("Error: 'data/cleaned_blogtext.csv' not found. Please run the save script first!")