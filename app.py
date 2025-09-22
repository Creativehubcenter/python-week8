"""
app.py
Streamlit dashboard for CORD-19 metadata
Run: streamlit run app.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set(style="whitegrid")

st.set_page_config(page_title="CORD-19 Explorer", layout="wide")

DEFAULT_CSV = "metadata.csv"

@st.cache_data
def load_data(path):
    """Try robust CSV loading, return DataFrame or None on failure."""
    try:
        df = pd.read_csv(path, low_memory=False)
        return df
    except pd.errors.ParserError:
        # retry with python engine and skip bad lines
        try:
            df = pd.read_csv(path, engine="python", low_memory=False, on_bad_lines="skip", encoding="utf-8")
            return df
        except Exception as e:
            st.error(f"Failed to parse CSV even with fallback: {e}")
            return None
    except FileNotFoundError:
        return None

def prepare_df(df):
    # select commonly useful columns if present
    cols = ["title", "abstract", "publish_time", "journal", "authors", "doi", "source_x"]
    keep = [c for c in cols if c in df.columns]
    df = df[keep].copy()
    df = df.dropna(subset=["title"])
    df["publish_time"] = pd.to_datetime(df.get("publish_time"), errors="coerce")
    df = df.dropna(subset=["publish_time"])
    df["year"] = df["publish_time"].dt.year
    df["abstract_length"] = df["abstract"].fillna("").apply(lambda x: len(x.split()))
    return df

def main():
    st.title("CORD-19 Research Dataset Explorer")

    # allow upload or default file
    st.sidebar.header("Data source")
    uploaded = st.sidebar.file_uploader("Upload metadata.csv (optional)", type=["csv"])
    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded, low_memory=False)
            st.sidebar.success("Uploaded file loaded.")
        except pd.errors.ParserError:
            df_raw = pd.read_csv(uploaded, engine="python", low_memory=False, on_bad_lines="skip")
            st.sidebar.warning("Loaded with fallback (skipped bad lines).")
    else:
        if os.path.exists(DEFAULT_CSV):
            df_raw = load_data(DEFAULT_CSV)
            if df_raw is None:
                st.error("metadata.csv exists but couldn't be parsed. Try uploading via sidebar.")
                return
        else:
            st.warning("No metadata.csv found in app folder. Upload a file via the sidebar.")
            df_raw = None

    if df_raw is None:
        st.info("Upload a CSV to begin exploration.")
        return

    df = prepare_df(df_raw)

    # Sidebar filters
    years = sorted(df["year"].dropna().unique().tolist())
    selected_year = st.sidebar.selectbox("Filter by Year (optional)", ["All"] + [str(y) for y in years])
    journals = df["journal"].dropna().unique().tolist()
    selected_journal = st.sidebar.selectbox("Filter by Journal (optional)", ["All"] + journals[:50])

    # apply filters
    filtered = df.copy()
    if selected_year != "All":
        filtered = filtered[filtered["year"] == int(selected_year)]
    if selected_journal != "All":
        filtered = filtered[filtered["journal"] == selected_journal]

    st.sidebar.markdown(f"**Records:** {len(filtered):,}")

    # main content
    st.subheader("Papers (table)")
    st.dataframe(filtered[["publish_time", "title", "journal", "authors"]].head(200), use_container_width=True)

    # Visualizations
    st.subheader("Visualizations")

    # Publications per year (global)
    pubs_by_year = df["year"].value_counts().sort_index()
    fig1, ax1 = plt.subplots(figsize=(8,4))
    pubs_by_year.plot(kind="line", marker="o", ax=ax1)
    ax1.set_title("Publications per Year")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

    # Top journals
    st.write("Top journals (global)")
    top_j = df["journal"].value_counts().head(15)
    fig2, ax2 = plt.subplots(figsize=(8,5))
    sns.barplot(x=top_j.values, y=top_j.index, ax=ax2)
    ax2.set_xlabel("Count")
    ax2.set_ylabel("Journal")
    st.pyplot(fig2)

    # Abstract length distribution for filtered set
    st.write("Abstract length distribution (filtered set)")
    fig3, ax3 = plt.subplots(figsize=(8,4))
    sns.histplot(filtered["abstract_length"], bins=40, kde=True, ax=ax3)
    ax3.set_xlabel("Abstract length (words)")
    st.pyplot(fig3)

    # Quick search in titles/abstracts
    st.subheader("Search papers")
    keyword = st.text_input("Search keyword (title or abstract)", "")
    if keyword:
        hits = df[df.apply(lambda r: keyword.lower() in str(r.get("title", "")).lower() or
                                       keyword.lower() in str(r.get("abstract", "")).lower(), axis=1)]
        st.write(f"Found {len(hits):,} papers containing '{keyword}'")
        st.dataframe(hits[["publish_time", "title", "journal"]].head(200), use_container_width=True)

    st.markdown("---")
    st.write("Notes:")
    st.write("- If CSV fails to parse, try uploading it via the sidebar (Streamlit will use a fallback parser).")
    st.write("- The 'journal' field can be noisy; some values are preprint servers or empty.")
    st.write("- For big datasets, the app may be slow—consider sampling or pre-processing.")

if __name__ == "__main__":
    main()
