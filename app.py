import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Fashion Trends Analysis",
    page_icon="👗",
    layout="wide"
)

st.title("👗 Fashion Trends Analysis Dashboard")
st.write("Explore fashion products, brands, categories, prices and discounts.")

st.divider()


# --------------------------------
# DATASET UPLOAD
# --------------------------------

file = st.file_uploader("Upload Fashion_Trends.csv", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    st.success("Dataset uploaded successfully!")

    # Convert numeric columns
    for col in ['Original_Price', 'Discount_Price', 'Discount_Percentage']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')


    # --------------------------------
    # SIDEBAR
    # --------------------------------

    st.sidebar.title("Fashion Analysis")
    menu = st.sidebar.selectbox(
        "Select Analysis",
        [
            "Dataset Overview",
            "Statistical Analysis",
            "Brand Analysis",
            "Category Analysis",
            "Gender Analysis",
            "Price Analysis",
            "Discount Analysis",
            "Price Segment Analysis",
            "Outlier Removal"
        ]
    )


    # --------------------------------
    # DATASET OVERVIEW
    # --------------------------------

    if menu == "Dataset Overview":

        st.header("Dataset Overview")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Missing Values", int(df.isnull().sum().sum()))

        st.subheader("Dataset")

        st.dataframe(df, use_container_width=True)

        st.subheader("Missing Values")

        st.dataframe(df.isnull().sum())

        st.subheader("Data Types")

        st.dataframe(df.dtypes.astype(str).to_frame("Data Type"))


    # --------------------------------
    # STATISTICAL ANALYSIS
    # --------------------------------

    elif menu == "Statistical Analysis":

        st.header("Statistical Analysis")

        st.subheader("Descriptive Statistics")

        st.dataframe(df.describe(), use_container_width=True)

        st.subheader("Category Count")

        st.dataframe(df['Category'].value_counts())

        st.subheader("Product Name Count")

        st.dataframe(df['Product_Name'].value_counts())

        st.subheader("Brand Count")

        st.dataframe(df['Brand'].value_counts())


    # --------------------------------
    # BRAND ANALYSIS
    # --------------------------------

    elif menu == "Brand Analysis":

        st.header("Brand Analysis")

        b = df['Brand'].value_counts()

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Top 10 Brands")

            fig, ax = plt.subplots()

            sb.barplot(x=b.nlargest(10).index,
                       y=b.nlargest(10).values,
                       ax=ax)

            plt.xticks(rotation=90)
            st.pyplot(fig)
            plt.close(fig)

        with col2:

            st.subheader("Bottom 10 Brands")

            fig, ax = plt.subplots()

            sb.barplot(x=b.nsmallest(10).index,
                       y=b.nsmallest(10).values,
                       ax=ax)

            plt.xticks(rotation=90)
            st.pyplot(fig)
            plt.close(fig)


    # --------------------------------
    # CATEGORY ANALYSIS
    # --------------------------------

    elif menu == "Category Analysis":

        st.header("Category Analysis")

        c = df['Category'].value_counts()

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Top 10 Categories")

            fig, ax = plt.subplots()

            sb.barplot(x=c.nlargest(10).index,
                       y=c.nlargest(10).values,
                       ax=ax)

            plt.xticks(rotation=90)
            st.pyplot(fig)
            plt.close(fig)

        with col2:

            st.subheader("Bottom 10 Categories")

            fig, ax = plt.subplots()

            sb.barplot(x=c.nsmallest(10).index,
                       y=c.nsmallest(10).values,
                       ax=ax)

            plt.xticks(rotation=90)
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Top 10 Categories by Product Count")

        category_count = df.groupby('Category')['Product_Name'].size().nlargest(10)

        st.dataframe(category_count.to_frame("Product Count"))


    # --------------------------------
    # GENDER ANALYSIS
    # --------------------------------

    elif menu == "Gender Analysis":

        st.header("Gender Analysis")

        g = df['Gender'].value_counts()

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Gender Distribution")

            fig, ax = plt.subplots()

            sb.barplot(x=g.index, y=g.values, ax=ax)

            plt.xticks(rotation=90)
            st.pyplot(fig)
            plt.close(fig)

        with col2:

            st.subheader("Gender-wise Product Count")

            gender_products = df.groupby('Gender')['Product_Name'].size()

            st.dataframe(gender_products.to_frame("Product Count"))

        st.subheader("Average Discount Percentage by Gender")

        st.dataframe(
            df.groupby('Gender')['Discount_Percentage'].mean().to_frame(
                "Average Discount"
            )
        )

        st.subheader("Gender vs Discount Percentage")

        fig, ax = plt.subplots()

        sb.boxplot(
            x=df['Gender'],
            y=df['Discount_Percentage'],
            ax=ax
        )

        st.pyplot(fig)
        plt.close(fig)


    # --------------------------------
    # PRICE ANALYSIS
    # --------------------------------

    elif menu == "Price Analysis":

        st.header("Price Analysis")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Original Price Distribution")

            fig, ax = plt.subplots()

            sb.histplot(df['Original_Price'].dropna(), ax=ax)

            st.pyplot(fig)
            plt.close(fig)

        with col2:

            st.subheader("Discount Price Distribution")

            fig, ax = plt.subplots()

            sb.histplot(df['Discount_Price'].dropna(), ax=ax)

            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Original Price Boxplot")

        fig, ax = plt.subplots()

        sb.boxplot(x=df['Original_Price'], ax=ax)

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Original Price vs Discount Price")

        fig, ax = plt.subplots()

        sb.scatterplot(
            x=df['Original_Price'],
            y=df['Discount_Price'],
            ax=ax
        )

        ax.set_title("Price VS Discount Price")

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Price vs Discount Percentage")

        fig, ax = plt.subplots()

        sb.scatterplot(
            x=df['Original_Price'],
            y=df['Discount_Percentage'],
            ax=ax
        )

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Price vs Discount Tier")

        fig, ax = plt.subplots()

        sb.stripplot(
            x=df['Original_Price'],
            y=df['Discount_Tier'],
            ax=ax
        )

        st.pyplot(fig)
        plt.close(fig)


    # --------------------------------
    # DISCOUNT ANALYSIS
    # --------------------------------

    elif menu == "Discount Analysis":

        st.header("Discount Analysis")

        st.subheader("Discount Tier Distribution")

        st.dataframe(df['Discount_Tier'].value_counts())

        st.subheader("Average Discount by Product")

        product_discount = df.groupby('Product_Name')['Discount_Percentage'].mean()

        st.dataframe(product_discount)

        st.subheader("Category vs Discount Tier")

        dt = pd.pivot_table(
            df,
            index='Category',
            columns='Discount_Tier',
            values='Product_ID',
            aggfunc='count',
            fill_value=0,
            margins=True
        )

        st.dataframe(dt)

        fig, ax = plt.subplots(figsize=(12, 6))

        sb.heatmap(dt.iloc[:-1, :-1], annot=True, fmt='g', ax=ax)

        plt.xticks(rotation=45)

        st.pyplot(fig)
        plt.close(fig)


    # --------------------------------
    # PRICE SEGMENT ANALYSIS
    # --------------------------------

    elif menu == "Price Segment Analysis":

        st.header("Price Segment Analysis")

        st.subheader("Price Segment Distribution")

        st.dataframe(df['Price_Segment'].value_counts())

        st.subheader("Category vs Price Segment")

        ct = pd.crosstab(
            df['Category'],
            df['Price_Segment']
        )

        st.dataframe(ct)

        fig, ax = plt.subplots(figsize=(12, 6))

        sb.heatmap(ct, annot=True, fmt='g', ax=ax)

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Brand vs Price Segment")

        pt = pd.pivot_table(
            df,
            index='Brand',
            columns='Price_Segment',
            values='Original_Price',
            aggfunc='mean'
        )

        st.dataframe(pt)

        fig, ax = plt.subplots(figsize=(12, 8))

        sb.heatmap(pt, cmap='coolwarm', ax=ax)

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Product Count by Price Segment")

        st.dataframe(
            df.groupby('Price_Segment')['Category'].size().to_frame(
                "Product Count"
            )
        )


    # --------------------------------
    # OUTLIER REMOVAL
    # --------------------------------

    elif menu == "Outlier Removal":

        st.header("Outlier Removal")

        clean_df = df.copy()

        for col in ['Original_Price', 'Discount_Price']:

            st.subheader(col + " - Before Outlier Removal")

            fig, ax = plt.subplots()

            sb.boxplot(x=clean_df[col], ax=ax)

            st.pyplot(fig)
            plt.close(fig)

            q1 = clean_df[col].quantile(0.25)
            q3 = clean_df[col].quantile(0.75)

            iqr = q3 - q1

            c1 = q1 - 1.5 * iqr
            c2 = q3 + 1.5 * iqr

            clean_df = clean_df[
                (clean_df[col] >= c1) &
                (clean_df[col] <= c2)
            ]

            st.subheader(col + " - After Outlier Removal")

            fig, ax = plt.subplots()

            sb.boxplot(x=clean_df[col], ax=ax)

            st.pyplot(fig)
            plt.close(fig)

        st.success("Outliers removed successfully!")

        col1, col2 = st.columns(2)

        col1.metric("Original Rows", len(df))
        col2.metric("Rows After Cleaning", len(clean_df))

        st.subheader("Cleaned Dataset")

        st.dataframe(clean_df, use_container_width=True)

        csv = clean_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download Cleaned Dataset",
            data=csv,
            file_name="Cleaned_Fashion_Trends.csv",
            mime="text/csv"
        )


else:

    st.info("Please upload your Fashion_Trends.csv file to start analysis.")

    st.write("Your Fashion Trends Analysis Dashboard will appear here.")
