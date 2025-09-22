"""
analysis.py
Basic exploration and visualization for CORD-19 metadata.csv
Run: python analysis.py
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

CSV_PATH = "metadata.csv"  # change if your file is elsewhere

def load_metadata(path):
    """Load metadata.csv with several fallbacks to avoid ParserError."""
    try:
        # primary attempt (fast)
        df = pd.read_csv(path, low_memory=False)
        print("Loaded with default engine.")
        return df
    except pd.errors.ParserError as e:
        print("ParserError with default engine:", e)
        print("Retrying with engine='python' and on_bad_lines='skip' ...")
        try:
            df = pd.read_csv(path, engine="python", low_memory=False, on_bad_lines="skip", encoding="utf-8")
            print("Loaded with python engine and skipped bad lines.")
            return df
        except Exception as e2:
            print("Failed to load CSV:", e2)
            raise

def clean_metadata(df):
    """Select useful columns and clean publish_time."""
    cols = ["title", "abstract", "publish_time", "journal", "authors", "doi", "source_x"]
    # keep columns that exist
    keep = [c for c in cols if c in df.columns]
    df = df[keep].copy()

    # drop rows without title since they're usually useless
    df = df.dropna(subset=["title"])

    # parse dates robustly
    df["publish_time"] = pd.to_datetime(df.get("publish_time"), errors="coerce")

    # drop rows without valid dates (if you need them keep this optional)
    df = df.dropna(subset=["publish_time"])

    # create helper columns
    df["year"] = df["publish_time"].dt.year
    df["abstract_length"] = df["abstract"].fillna("").apply(lambda x: len(x.split()))
    df["title_length"] = df["title"].apply(lambda x: len(str(x).split()))
    return df

def basic_stats(df):
    print("\n--- Basic stats for numeric columns ---")
    print(df[["abstract_length", "title_length"]].describe())

    print("\n--- Top journals ---")
    print(df["journal"].value_counts().head(15))

    print("\n--- Publications per year ---")
    pubs = df["year"].value_counts().sort_index()
    print(pubs)

    return pubs

def save_plot_fig(fig, filename):
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"Saved plot: {filename}")

def create_visualizations(df, pubs_by_year):
    # 1. Publications over time (line)
    fig1, ax1 = plt.subplots(figsize=(10,5))
    pubs_by_year.plot(kind="line", marker="o", ax=ax1)
    ax1.set_title("Publications per Year (CORD-19 metadata)")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of Publications")
    save_plot_fig(fig1, "publications_per_year.png")

    # 2. Top journals (bar)
    top_j = df["journal"].value_counts().head(15)
    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.barplot(x=top_j.values, y=top_j.index, ax=ax2)
    ax2.set_title("Top 15 Journals (by count)")
    ax2.set_xlabel("Number of Papers")
    ax2.set_ylabel("Journal")
    save_plot_fig(fig2, "top_journals.png")

    # 3. Abstract length distribution (hist)
    fig3, ax3 = plt.subplots(figsize=(10,5))
    sns.histplot(df["abstract_length"], bins=50, kde=True, ax=ax3)
    ax3.set_title("Distribution of Abstract Lengths (words)")
    ax3.set_xlabel("Abstract length (words)")
    save_plot_fig(fig3, "abstract_length_distribution.png")

    # 4. Scatter: title length vs abstract length (sample)
    # sample if too large
    sample = df.sample(n=min(5000, len(df)), random_state=1)
    fig4, ax4 = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=sample, x="title_length", y="abstract_length", hue="year", palette="viridis", legend=False, ax=ax4)
    ax4.set_title("Title length vs Abstract length (sample)")
    ax4.set_xlabel("Title length (words)")
    ax4.set_ylabel("Abstract length (words)")
    save_plot_fig(fig4, "title_vs_abstract.png")

def observations(df):
    print("\n--- Observations & Notes ---")
    print("- Set of columns available may vary depending on metadata version.")
    print("- Many rows may have missing abstracts; we filled length with 0 for those.")
    print("- Publications timeline shows spikes depending on date coverage in metadata.")
    print("- Top journals include both preprint servers and journals; 'journal' field can be noisy.")
    print("\nSuggestions:")
    print(" - If dataset is large, consider processing in chunks or using Dask.")
    print(" - Use text cleaning (remove punctuation, stopwords) before deep text analysis.")
    print(" - For assignment, include screenshots of output plots and a short README.")

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found in current directory: {os.getcwd()}")
        print("Place metadata.csv in this folder or update CSV_PATH in the script.")
        sys.exit(1)

    df_raw = load_metadata(CSV_PATH)
    df = clean_metadata(df_raw)

    print(f"\nCleaned dataset shape: {df.shape}")
    pubs_by_year = basic_stats(df)

    create_visualizations(df, pubs_by_year)
    observations(df)
    print("\nDone. Plots saved as PNG files in current directory.")

if __name__ == "__main__":
    main()
