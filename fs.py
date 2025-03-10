# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 09:50:01 2025

@author: ConnorChristensen
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

file_path = "C:/Users/ConnorChristensen/OneDrive - Wyoming Business Council/Documents/GitHub/Fiscal-System-WY/WYSalesUse22_23_24_.csv"  
df = pd.read_csv(file_path)

# Convert all numeric columns to proper types
for col in df.columns[2:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Set seaborn theme
sns.set_theme(style="whitegrid")

### 1. Bar Chart: Total Sales by County Over the Years
plt.figure(figsize=(14, 7))
df_grouped = df.groupby("County").sum(numeric_only=True)
df_grouped[['2022 Total Sales', '2023 Total Sales', '2024 Total Use Tax']].plot(kind='bar', stacked=True, figsize=(14, 7))
plt.title("Total Sales by County (2022-2024)")
plt.xlabel("County")
plt.ylabel("Total Sales ($)")
plt.legend(title="Year")
plt.xticks(rotation=90)
plt.show()

### 2. Boxplot: Year-over-Year Growth Rate Comparison
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['2023 Total Sales YoY Growth Rate (%)', '2024 Total Use YoY Growth Rate (%)']])
plt.title("Year-over-Year Growth Rate Distribution for Total Sales")
plt.ylabel("Growth Rate (%)")
plt.xticks(ticks=[0, 1], labels=['2023', '2024'])
plt.show()

### 3. Scatter Plot: Total Sales vs Use Tax (2023 vs 2024)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df["2023 Total Sales"], y=df["2024 Total Use Tax"], hue=df["County"], alpha=0.7)
plt.title("Total Sales (2023) vs. Use Tax (2024)")
plt.xlabel("Total Sales (2023)")
plt.ylabel("Total Use Tax (2024)")
plt.legend(bbox_to_anchor=(1, 1))
plt.show()

### 4. Heatmap: Correlation Between Financial Metrics
plt.figure(figsize=(10, 8))
sns.heatmap(df_grouped.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap Between Sales & Tax Variables")
plt.show()

### 5. Line Chart: Industry-wise Sales Trend Over Time
plt.figure(figsize=(12, 7))
industries = df["Industry"].unique()[:10]  # Limit to first 10 industries for clarity
for industry in industries:
    industry_data = df[df["Industry"] == industry]
    plt.plot(["2022", "2023", "2024"], 
             [industry_data["2022 Total Sales"].sum(), 
              industry_data["2023 Total Sales"].sum(), 
              industry_data["2024 Total Use Tax"].sum()], 
             marker='o', label=industry)

plt.title("Industry-wise Total Sales Trend (2022-2024)")
plt.xlabel("Year")
plt.ylabel("Total Sales ($)")
plt.legend(title="Industry", bbox_to_anchor=(1, 1))
plt.show()

top_growth = df.groupby("County")[["2023 Total Sales YoY Growth Rate (%)", "2024 Total Use YoY Growth Rate (%)"]].mean()
top_growth.sort_values(by="2024 Total Use YoY Growth Rate (%)", ascending=False)

###########broken
df_grouped = df.groupby(["County", "Industry"])[["2022 Total Sales", "2023 Total Sales", "2024 Total Use Tax"]].sum()
df_grouped["Percentage"] = df_grouped.groupby(level=0).apply(lambda x: x / x.sum())
df_grouped.reset_index().sort_values("Percentage", ascending=False)
###############

#################broken
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

county_features = df.groupby("County")[["2023 Total Sales", "2024 Total Use Tax"]].mean()
scaler = StandardScaler()
county_features_scaled = scaler.fit_transform(county_features)

kmeans = KMeans(n_clusters=4, random_state=42)
county_features["Cluster"] = kmeans.fit_predict(county_features_scaled)
county_features

pip install sklearn
#################################


industry_volatility = df.groupby("Industry")[["2023 Total Sales YoY Growth Rate (%)", "2024 Total Use YoY Growth Rate (%)"]].std()
industry_volatility.sort_values(by="2024 Total Use YoY Growth Rate (%)", ascending=False)


industry_trends = df.groupby("Industry")[["2022 Total Sales", "2023 Total Sales", "2024 Total Use Tax"]].mean()
industry_trends["YoY Growth (22-23)"] = (industry_trends["2023 Total Sales"] - industry_trends["2022 Total Sales"]) / industry_trends["2022 Total Sales"] * 100
industry_trends["YoY Growth (23-24)"] = (industry_trends["2024 Total Use Tax"] - industry_trends["2023 Total Sales"]) / industry_trends["2023 Total Sales"] * 100
industry_trends.sort_values(by="YoY Growth (23-24)", ascending=False)



#############################################################################################################




import seaborn as sns
import matplotlib.pyplot as plt

industry_correlation = df.groupby("Industry").sum(numeric_only=True).corr()
plt.figure(figsize=(10, 6))
sns.heatmap(industry_correlation, annot=True, cmap="coolwarm")
plt.title("Industry Correlation Heatmap")
plt.show()



df_tax = df.groupby("County")[["2024 Total Use Tax", "2024 4% Use Tax"]].sum()
df_tax["Use Tax Percentage"] = df_tax["2024 Total Use Tax"] / df_tax["2024 4% Use Tax"] * 100
df_tax.sort_values(by="Use Tax Percentage", ascending=False)


#broken#############################
from statsmodels.tsa.holtwinters import ExponentialSmoothing

df_forecast = df.groupby("Fiscal Year")[["2022 Total Sales", "2023 Total Sales", "2024 Total Use Tax"]].sum()
model = ExponentialSmoothing(df_forecast["2024 Total Use Tax"], trend="add", seasonal="add", seasonal_periods=3).fit()
df_forecast["2025 Prediction"] = model.forecast(1)
df_forecast
#########################################

#################################broken
df_policy = df[df["County"].isin(["Teton", "Laramie", "Natrona"])]  # Example counties with tax policy changes
df_policy.groupby("Fiscal Year")[["2023 Total Sales", "2024 Total Use Tax"]].sum().plot(kind='line', figsize=(10, 6))
plt.title("Effect of Tax Policy Changes on Revenue")
plt.xlabel("Year")
plt.ylabel("Revenue ($)")
plt.legend(title="Tax Category")
plt.show()
#################################################

df_recovery = df.groupby("Industry")[["2023 Total Sales YoY Growth Rate (%)", "2024 Total Use YoY Growth Rate (%)"]].apply(lambda x: x - x.mean())
df_recovery.sort_values(by="2024 Total Use YoY Growth Rate (%)", ascending=False)


#######################################################################################################################





