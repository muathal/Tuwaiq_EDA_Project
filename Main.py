import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE SETUP ---
st.set_page_config(page_title="Blog Analysis Dashboard", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_blogtext.csv")


try:
    ndf = load_data()
    st.title("📝 Blog Authorship Analysis Dashboard")

    # --- SIDEBAR: TOP USERS ---
    st.sidebar.header("🏆 Most Active Users")
    st.sidebar.write("There are common users who have been highly active:")
    st.sidebar.info("- **User 449628**: 206 posts\n- **User 589736**: 130 posts")

    # Show Top 5 Table in Sidebar
    top_5 = ndf["id"].value_counts().head(5)
    st.sidebar.table(top_5)

    # --- TABBED ANALYSIS ---
    tab1, tab2, tab3 = st.tabs(["Demographics", "Topic Analysis", "Trends"])

    with tab1:
        st.header("User Demographics")

        # FINDINGS 1 & 2
        st.success(
            "**Findings:**\n* Female and male distribution is **very close** to each other.\n* Most users are **Young Adults**, followed by **Teens**.")

        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots()
            sns.countplot(data=ndf, x="gender", ax=ax1, palette="pastel")
            ax1.set_title("Gender Distribution")
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots()
            sns.countplot(data=ndf, x="age group", order=["Teen", "Young Adult", "Adult", "Old"], ax=ax2,
                          palette="viridis")
            ax2.set_title("Age Group Distribution")
            st.pyplot(fig2)

    with tab2:
        st.header("Topic & Industry Insights")

        # FINDINGS 3, 4, 5, 6
        st.info("""
        **Findings:**
        * Most text falls under **indUnk** (Industry Unknown), followed by the **Student** topic.
        * Female and Male are **equal** in the Student topic.
        * **Males** are more dominant in: *Technology, Religion, and Internet*.
        * **Females** are more dominant in: *Fashion, Arts, and Education*.
        """)

        st.subheader("Distribution by Topic")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.countplot(data=ndf, y="topic", order=ndf["topic"].value_counts().index, ax=ax3)
        st.pyplot(fig3)

        st.subheader("Topic Distribution by Gender")
        fig4, ax4 = plt.subplots(figsize=(10, 8))
        sns.countplot(data=ndf, y="topic", hue="gender", ax=ax4)
        st.pyplot(fig4)

    with tab3:
        st.header("Activity Trends")

        # FINDING 7
        st.warning(
            "**Finding:** There is an **increasing trend** in foreigners (non-English users) using the website over the years.")

        st.subheader("Non-English Usage Over Time")
        # Ensure year is treated correctly
        non_en_data = ndf[ndf["Lang"] == "non-en"].groupby("year").size()
        st.line_chart(non_en_data)

except FileNotFoundError:
    st.error("Please ensure 'cleaned_blogtext.csv' exists in the directory.")