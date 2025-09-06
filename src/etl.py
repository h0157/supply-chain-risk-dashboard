import pandas as pd
import requests
import time
from textblob import TextBlob
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# =============================
# Step 1 - Load Historical Data
# =============================
supplier_df = pd.read_csv("data/raw/historical_supplier_data.csv")
logistics_df = pd.read_csv("data/raw/logistics_route_data.csv")

# =============================
# Step 2 - Generalized Cleaning Functions
# =============================
def clean_data(df, numeric_columns=None, categorical_columns=None, date_columns=None):
    """Generalized function to clean data."""
    df = df.copy()
    logging.info(f"Starting cleaning for dataset with {len(df)} rows.")

    # Remove duplicates
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    logging.info(f"Removed {initial_rows - len(df)} duplicate rows.")
   # Handle missing values
    if numeric_columns:
        for col in numeric_columns:
            if col in df.columns:
                median = df[col].median()
                df[col] = df[col].fillna(median)
                logging.info(f"Filled missing values in numeric column '{col}' with median: {median}.")

    if categorical_columns:
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")
                logging.info(f"Filled missing values in categorical column '{col}' with 'Unknown'.")

    if date_columns:
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df[col] = df[col].fillna(pd.Timestamp("1970-01-01"))
                logging.info(f"Converted '{col}' to datetime and filled invalid dates with '1970-01-01'.")

    # Standardize categorical data
    if categorical_columns:
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].str.strip().str.title()
                logging.info(f"Standardized text in categorical column '{col}'.")

    # Validate data types
    if numeric_columns:
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
                logging.info(f"Validated and converted '{col}' to numeric.")

    logging.info("Data cleaning completed.")
    return df

def handle_outliers(df, column):
    """Cap outliers using the IQR method."""
    if column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        logging.info(f"Detected {len(outliers)} outliers in column '{column}'.")
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        logging.info(f"Capped outliers in column '{column}' to range [{lower_bound}, {upper_bound}].")
    return df

def standardize_countries(df, column):
    """Standardize country names using a mapping dictionary."""
    country_mapping = {
        "Usa": "United States",
        "Uk": "United Kingdom",
        "Uae": "United Arab Emirates",
    }
    if column in df.columns:
        df[column] = df[column].replace(country_mapping)
        logging.info(f"Standardized country names in column '{column}'.")
    return df

# =============================
# Step 3 - Apply Cleaning Functions
# =============================
supplier_df = clean_data(
    supplier_df,
    numeric_columns=["on_time_delivery"],
    categorical_columns=["supplier_name"],
    date_columns=["order_date"]
)
logistics_df = clean_data(
    logistics_df,
    numeric_columns=["transit_time_days"],
        categorical_columns=["origin_country", "destination_country"]
)

# Handle outliers
supplier_df = handle_outliers(supplier_df, "on_time_delivery")
logistics_df = handle_outliers(logistics_df, "transit_time_days")

# Standardize country names
logistics_df = standardize_countries(logistics_df, "origin_country")
logistics_df = standardize_countries(logistics_df, "destination_country")

# =============================
# Step 4 - Validate Data Ranges
# =============================
def validate_data(df, column, valid_range=None):
    """Validate data in a column and replace invalid values."""
    if valid_range and column in df.columns:
        invalid_count = (~df[column].between(valid_range[0], valid_range[1])).sum()
        df.loc[~df[column].between(valid_range[0], valid_range[1]), column] = valid_range[0]
        logging.info(f"Validated column '{column}' and replaced {invalid_count} invalid values.")
    return df

logistics_df = validate_data(logistics_df, "transit_time_days", valid_range=(0, 365))

# =============================
# Step 5 - Save Cleaned Data
# =============================
supplier_df.to_csv("data/processed/cleaned_supplier_data.csv", index=False)
logistics_df.to_csv("data/processed/cleaned_logistics_data.csv", index=False)
logging.info("Cleaned data saved to 'data/processed/'.")

# =============================
# Step 6 - Weather API Setup
# =============================
WEATHER_API_KEY = "bd0ed3042bc7862dd5c143e7a31488a5"
WEATHER_URL = "http://api.weatherapi.com/v1/current.json"

def get_weather_for_country(country):
    try:
        response = requests.get(WEATHER_URL, params={
            "key": WEATHER_API_KEY,
            "q": country
        })
        data = response.json()
        condition = data["current"]["condition"]["text"]
        temp_c = data["current"]["temp_c"]

        if "storm" in condition.lower() or "rain" in condition.lower():
            weather_risk = 0.8
        elif "snow" in condition.lower():
            weather_risk = 0.7
        else:
            weather_risk = 0.2
        return {"country": country, "weather_condition": condition,
                "temperature_c": temp_c, "weather_risk": weather_risk}
    except Exception as e:
        return {"country": country, "weather_condition": "Unknown",
                "temperature_c": None, "weather_risk": 0.5}

# =============================
# Step 7 - News API Setup
# =============================
NEWS_API_KEY = "9e052a18d3594955bc8299e0857506b6"
NEWS_URL = "https://newsapi.org/v2/top-headlines"

def get_news_sentiment_for_country(country):
    try:
        response = requests.get(NEWS_URL, params={
            "apiKey": NEWS_API_KEY,
            "q": country,
            "language": "en",
            "pageSize": 3
        })
        articles = response.json().get("articles", [])
        if not articles:
            return {"country": country, "news_sentiment": 0.0}

        sentiments = []
        for article in articles:
            text = article["title"] + " " + article.get("description", "")
            polarity = TextBlob(text).sentiment.polarity
            sentiments.append(polarity)

        avg_sentiment = sum(sentiments) / len(sentiments)
        # Convert sentiment to risk: negative → higher risk
        news_risk = 1 - ((avg_sentiment + 1) / 2)
        return {"country": country, "news_sentiment": avg_sentiment, "news_risk": news_risk}
    except Exception as e:
        return {"country": country, "news_sentiment": 0.0, "news_risk": 0.5}

# =============================
# Step 8 - Collect Real-Time Data
# =============================
unique_countries = logistics_df["origin_country"].unique()

weather_data = []
news_data = []

for country in unique_countries:
    print(f"Fetching weather & news for {country}...")
    weather_data.append(get_weather_for_country(country))
    news_data.append(get_news_sentiment_for_country(country))
    time.sleep(1)  # prevent API rate limit

weather_df = pd.DataFrame(weather_data)
news_df = pd.DataFrame(news_data)

# =============================
# Step 9 - Merge All Data
# =============================
merged = logistics_df.merge(weather_df, left_on="origin_country", right_on="country", how="left")
merged = merged.merge(news_df, left_on="origin_country", right_on="country", how="left")

# Drop duplicate "country" columns
merged.drop(columns=["country_x", "country_y"], inplace=True, errors="ignore")

# =============================
# Step 10 - Save Processed Data
# =============================
merged.to_csv("data/processed/logistics_with_realtime.csv", index=False)
print("Saved to data/processed/logistics_with_realtime.csv")


