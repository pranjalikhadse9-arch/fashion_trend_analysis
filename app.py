import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="Fashion Trends Analysis",
    page_icon="👗",
    layout="wide"
)

# Title
st.title("👗 Fashion Trends Analysis Dashboard")
st.write("Explore fashion products, brands, prices, discounts and categories.")

# Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("Fashion_Trends.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset not found. Please check the CSV file path.")
    st.stop()

# Check required columns
required_columns = [
    "Original_Price", "Discount_Price", "Brand",
    "Category", "Gender", "Product_Name",
    "Price_Segment", "Discount_Tier",
    "Discount_Percentage", "Product_ID"
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"Missing columns: {missing_columns}")
    st.stop()

# Sidebar
st.sidebar.title("Fashion Analysis")
menu = st.sidebar.radio(
    "Select Analysis",
    [
        "Dataset Overview",
        "Univariate Analysis",
        "Outlier Treatment",
        "Discount Analysis",
        "Bivariate Analysis",
        "Price Segment Analysis"
    ]
)

# Function for charts
def show_chart(fig):
    st.pyplot(fig)
    plt.close(fig)


# Dataset Overview
if menu == "Dataset Overview":

    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Products", len(df))
    col2.metric("Total Brands", df["Brand"].nunique())
    col3.metric("Total Categories", df["Category"].nunique())

    st.subheader("Dataset")

    st.dataframe(df, use_container_width=True)

    st.subheader("Missing Values")

    st.dataframe(
        df.isnull().sum().reset_index().rename(
            columns={"index": "Column", 0: "Missing Values"}
        ),
        use_container_width=True
    )

    st.subheader("Data Types")
    st.dataframe(
        df.dtypes.astype(str).reset_index().rename(
            columns={"index": "Column", 0: "Data Type"}
        )
    )

    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Category Count")
    st.dataframe(df["Category"].value_counts())

    st.subheader("Product Name Count")
    st.dataframe(df["Product_Name"].value_counts())

    st.subheader("Brand Count")
    st.dataframe(df["Brand"].value_counts())


# Univariate Analysis
elif menu == "Univariate Analysis":

    st.header("Univariate Analysis")

    st.subheader("Original Price Distribution")

    fig, ax = plt.subplots()
    sns.histplot(df["Original_Price"].dropna(), ax=ax)
    ax.set_title("Original Price Distribution")
    show_chart(fig)

    st.subheader("Discount Price Distribution")

    fig, ax = plt.subplots()
    sns.histplot(df["Discount_Price"].dropna(), ax=ax)
    ax.set_title("Discount Price Distribution")
    show_chart(fig)

    # Top and Bottom 10 Brands
    st.subheader("Top 10 Brands")

    b = df["Brand"].value_counts().nlargest(10)

    fig, ax = plt.subplots()
    sns.barplot(x=b.index, y=b.values, ax=ax)
    ax.set_xticks(range(len(b)))
    ax.set_xticklabels(b.index, rotation=90)
    ax.set_title("Top 10 Brands")
    show_chart(fig)

    st.subheader("Bottom 10 Brands")

    b = df["Brand"].value_counts().nsmallest(10)

    fig, ax = plt.subplots()
    sns.barplot(x=b.index, y=b.values, ax=ax)
    ax.set_xticks(range(len(b)))
    ax.set_xticklabels(b.index, rotation=90)
    ax.set_title("Bottom 10 Brands")
    show_chart(fig)

    # Category
    st.subheader("Top 10 Categories")

    c = df["Category"].value_counts().nlargest(10)

    fig, ax = plt.subplots()
    sns.barplot(x=c.index, y=c.values, ax=ax)
    ax.set_xticklabels(c.index, rotation=90)
    ax.set_title("Top 10 Categories")
    show_chart(fig)

    st.subheader("Bottom 10 Categories")

    c = df["Category"].value_counts().nsmallest(10)

    fig, ax = plt.subplots()
    sns.barplot(x=c.index, y=c.values, ax=ax)
    ax.set_xticklabels(c.index, rotation=90)
    ax.set_title("Bottom 10 Categories")
    show_chart(fig)

    # Gender
    st.subheader("Gender Distribution")

    g = df["Gender"].value_counts()

    fig, ax = plt.subplots()
    sns.barplot(x=g.index, y=g.values, ax=ax)
    ax.set_title("Gender Distribution")
    show_chart(fig)


# Outlier Treatment
elif menu == "Outlier Treatment":

    st.header("Outlier Detection and Treatment")

    clean_df = df.copy()

    for column in ["Original_Price", "Discount_Price"]:

        st.subheader(f"{column} - Before Outlier Removal")

        fig, ax = plt.subplots()
        sns.boxplot(x=clean_df[column], ax=ax)
        show_chart(fig)

        q1 = clean_df[column].quantile(0.25)
        q3 = clean_df[column].quantile(0.75)

        iqr = q3 - q1

        c1 = q1 - 1.5 * iqr
        c2 = q3 + 1.5 * iqr

        outliers = clean_df[
            (clean_df[column] > c2) |
            (clean_df[column] < c1)
        ]

        st.write("Number of Outliers:", len(outliers))

        # Remove outliers
        clean_df = clean_df[
            (clean_df[column] >= c1) &
            (clean_df[column] <= c2)
        ]

        st.subheader(f"{column} - After Outlier Removal")

        fig, ax = plt.subplots()
        sns.boxplot(x=clean_df[column], ax=ax)
        show_chart(fig)

    st.success("Outlier treatment completed!")

    col1, col2 = st.columns(2)

    col1.metric("Original Rows", len(df))
    col2.metric("Rows After Cleaning", len(clean_df))

    st.subheader("Cleaned Dataset")
    st.dataframe(clean_df, use_container_width=True)

    st.download_button(
        "Download Cleaned Dataset",
        data=clean_df.to_csv(index=False),
        file_name="Fashion_Trends_Cleaned.csv",
        mime="text/csv"
    )


# Discount Analysis
elif menu == "Discount Analysis":

    st.header("Discount Analysis")

    st.subheader("Brand Count - Top 10")

    b = df["Brand"].value_counts().nlargest(10)
    st.dataframe(b)

    st.subheader("Category Count")

    c = df["Category"].value_counts()

    fig, ax = plt.subplots()
    sns.barplot(x=c.index, y=c.values, ax=ax)
    ax.set_xticklabels(c.index, rotation=90)
    show_chart(fig)

    st.subheader("Price Segment vs Category")

    st.dataframe(
        df.groupby("Price_Segment")["Category"].size()
    )

    st.subheader("Product Name Count")

    st.dataframe(df["Product_Name"].value_counts())

    st.subheader("Gender vs Product Count")

    st.dataframe(
        df.groupby("Gender")["Product_Name"].size()
    )

    st.subheader("Top 10 Categories by Product Count")

    category_count = (
        df.groupby("Category")["Product_Name"]
        .size()
        .nlargest(10)
    )

    st.dataframe(category_count)

    st.subheader("Discount Tier Count")

    st.dataframe(df.groupby("Discount_Tier").size())

    st.subheader("Average Discount Percentage by Product")

    st.dataframe(
        df.groupby("Product_Name")["Discount_Percentage"].mean()
    )

    st.subheader("Category vs Discount Tier")

    dt = pd.pivot_table(
        df,
        index="Category",
        columns="Discount_Tier",
        values="Product_ID",
        aggfunc="count",
        margins=True
    )

    st.dataframe(dt, use_container_width=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(dt, annot=True, fmt=".0f", ax=ax)
    ax.set_title("Category vs Discount Tier")
    show_chart(fig)


# Bivariate Analysis
elif menu == "Bivariate Analysis":

    st.header("Bivariate Analysis")

    st.subheader("Original Price vs Discount Price")

    fig, ax = plt.subplots()
    sns.scatterplot(
        x=df["Original_Price"],
        y=df["Discount_Price"],
        ax=ax
    )
    ax.set_title("Original Price vs Discount Price")
    show_chart(fig)

    st.subheader("Original Price vs Discount Tier")

    fig, ax = plt.subplots()
    sns.stripplot(
        x=df["Discount_Tier"],
        y=df["Original_Price"],
        ax=ax
    )
    ax.set_title("Original Price vs Discount Tier")
    show_chart(fig)

    st.subheader("Original Price vs Discount Percentage")

    fig, ax = plt.subplots()
    sns.scatterplot(
        x=df["Original_Price"],
        y=df["Discount_Percentage"],
        ax=ax
    )
    ax.set_title("Original Price vs Discount Percentage")
    show_chart(fig)

    st.subheader("Gender vs Category Count")

    st.dataframe(
        df.groupby("Gender")["Category"].count()
    )

    st.subheader("Gender vs Average Discount Percentage")

    st.dataframe(
        df.groupby("Gender")["Discount_Percentage"].mean()
    )

    st.subheader("Gender vs Discount Percentage")

    fig, ax = plt.subplots()
    sns.boxplot(
        x=df["Gender"],
        y=df["Discount_Percentage"],
        ax=ax
    )
    ax.set_title("Gender vs Discount Percentage")
    show_chart(fig)


# Price Segment Analysis
elif menu == "Price Segment Analysis":

    st.header("Price Segment Analysis")

    st.subheader("Price Segment Distribution")

    st.dataframe(df["Price_Segment"].value_counts())

    st.subheader("Category vs Price Segment")

    ct = pd.crosstab(
        df["Category"],
        df["Price_Segment"]
    )

    st.dataframe(ct, use_container_width=True)

    st.subheader("Brand vs Price Segment Heatmap")

    pt = pd.pivot_table(
        df,
        index="Brand",
        columns="Price_Segment",
        values="Original_Price",
        aggfunc="mean",
        margins=True
    )

    st.dataframe(pt, use_container_width=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        pt,
        annot=True,
        fmt=".1f",
        cmap="coolwarm",
        ax=ax
    )

    ax.set_title("Brand vs Price Segment")
    show_chart(fig)


# Footer
st.sidebar.markdown("---")
st.sidebar.write("Fashion Trends Analysis")
st.sidebar.caption("Created using Python & Streamlit")
