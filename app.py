
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="Fashion Trends Analysis",
    page_icon="👗",
    layout="wide"
)

st.title("Fashion Trends Analysis")
st.write("Data Analysis using Python and Streamlit")

# Load Dataset
st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload Fashion_Trends.csv",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    try:
        df = pd.read_csv("Fashion_Trends.csv")
    except FileNotFoundError:
        st.warning("Please upload Fashion_Trends.csv")
        st.stop()

# Sidebar Menu
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Select Analysis",
    [
        "Dataset Overview",
        "Univariate Analysis",
        "Outlier Treatment",
        "Category Analysis",
        "Discount Analysis",
        "Bivariate Analysis",
        "Price Segment Analysis"
    ]
)

# Create Cleaned Dataset
clean_df = df.copy()

for i in ["Original_Price", "Discount_Price"]:
    q1 = clean_df[i].quantile(0.25)
    q3 = clean_df[i].quantile(0.75)
    iqr = q3 - q1

    c1 = q1 - 1.5 * iqr
    c2 = q3 + 1.5 * iqr

    clean_df = clean_df[
        (clean_df[i] >= c1) &
        (clean_df[i] <= c2)
    ]

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
    st.dataframe(df.isnull().sum())

    st.subheader("Data Types")
    st.dataframe(df.dtypes.astype(str))

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

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
    sb.histplot(df["Original_Price"].dropna(), ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Discount Price Distribution")

    fig, ax = plt.subplots()
    sb.histplot(df["Discount_Price"].dropna(), ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Top 10 Brands")

    b = df["Brand"].value_counts().nlargest(10)

    fig, ax = plt.subplots()
    sb.barplot(x=b.index, y=b.values, ax=ax)
    ax.tick_params(axis="x", rotation=90)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Bottom 10 Brands")

    b = df["Brand"].value_counts().nsmallest(10)

    fig, ax = plt.subplots()
    sb.barplot(x=b.index, y=b.values, ax=ax)
    ax.tick_params(axis="x", rotation=90)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Top 10 Categories")

    c = df["Category"].value_counts().nlargest(10)

    fig, ax = plt.subplots()
    sb.barplot(x=c.index, y=c.values, ax=ax)
    ax.tick_params(axis="x", rotation=90)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Bottom 10 Categories")

    c = df["Category"].value_counts().nsmallest(10)

    fig, ax = plt.subplots()
    sb.barplot(x=c.index, y=c.values, ax=ax)
    ax.tick_params(axis="x", rotation=90)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Gender Distribution")

    g = df["Gender"].value_counts()

    fig, ax = plt.subplots()
    sb.barplot(x=g.index, y=g.values, ax=ax)
    st.pyplot(fig)
    plt.close(fig)


# Outlier Treatment
elif menu == "Outlier Treatment":

    st.header("Outlier Detection and Removal")

    for i in ["Original_Price", "Discount_Price"]:

        st.subheader(i + " - Before Outliers")

        fig, ax = plt.subplots()
        sb.boxplot(x=df[i], ax=ax)
        st.pyplot(fig)
        plt.close(fig)

        q1 = df[i].quantile(0.25)
        q3 = df[i].quantile(0.75)
        iqr = q3 - q1

        c1 = q1 - 1.5 * iqr
        c2 = q3 + 1.5 * iqr

        outliers = df[
            (df[i] < c1) | (df[i] > c2)
        ]

        st.write("Number of Outliers:", len(outliers))

        st.subheader(i + " - After Outliers")

        fig, ax = plt.subplots()
        sb.boxplot(x=clean_df[i], ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    st.success("Outlier Treatment Completed")

    col1, col2 = st.columns(2)

    col1.metric("Before Cleaning", len(df))
    col2.metric("After Cleaning", len(clean_df))

    st.subheader("Cleaned Dataset")

    st.dataframe(clean_df, use_container_width=True)

    st.download_button(
        "Download Cleaned Dataset",
        data=clean_df.to_csv(index=False),
        file_name="Cleaned_Fashion_Trends.csv",
        mime="text/csv"
    )


# Category Analysis
elif menu == "Category Analysis":

    st.header("Category Analysis")

    st.subheader("Top 10 Brands")
    st.dataframe(
        df["Brand"].value_counts().nlargest(10)
    )

    st.subheader("Category Count")

    c = df["Category"].value_counts()

    fig, ax = plt.subplots()
    sb.barplot(x=c.index, y=c.values, ax=ax)
    ax.tick_params(axis="x", rotation=90)
    st.pyplot(fig)
    plt.close(fig)

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

    st.dataframe(
        df.groupby("Category")["Product_Name"]
        .size()
        .nlargest(10)
    )


# Discount Analysis
elif menu == "Discount Analysis":

    st.header("Discount Analysis")

    st.subheader("Discount Tier Count")

    st.dataframe(
        df.groupby("Discount_Tier").size()
    )

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
        aggfunc="sum",
        margins=True
    )

    st.dataframe(dt, use_container_width=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    sb.barplot(data=dt, ax=ax)
    ax.tick_params(axis="x", rotation=90)
    st.pyplot(fig)
    plt.close(fig)


# Bivariate Analysis
elif menu == "Bivariate Analysis":

    st.header("Bivariate Analysis")

    st.subheader("Original Price vs Discount Price")

    fig, ax = plt.subplots()
    sb.scatterplot(
        x=df["Original_Price"],
        y=df["Discount_Price"],
        ax=ax
    )
    ax.set_title("Price VS Discount_Price")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Original Price vs Discount Tier")

    fig, ax = plt.subplots()
    sb.stripplot(
        x=df["Discount_Tier"],
        y=df["Original_Price"],
        ax=ax
    )
    ax.set_title("Price VS Discount_Tier")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Original Price vs Discount Percentage")

    fig, ax = plt.subplots()
    sb.scatterplot(
        x=df["Original_Price"],
        y=df["Discount_Percentage"],
        ax=ax
    )
    ax.set_title("Price VS Discount_Percentage")
    st.pyplot(fig)
    plt.close(fig)

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
    sb.boxplot(
        x=df["Gender"],
        y=df["Discount_Percentage"],
        ax=ax
    )
    st.pyplot(fig)
    plt.close(fig)


# Price Segment Analysis
elif menu == "Price Segment Analysis":

    st.header("Price Segment Analysis")

    st.subheader("Price Segment Count")

    st.dataframe(
        df["Price_Segment"].value_counts()
    )

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
        margins=True
    )

    st.dataframe(pt, use_container_width=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    sb.heatmap(pt, annot=True, ax=ax)
    st.pyplot(fig)
    plt.close(fig)


st.sidebar.markdown("---")
st.sidebar.write("Fashion Trends Analysis")
st.sidebar.caption("Python | Pandas | Seaborn | Streamlit")
