#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load IKEA datasets
ikea1 = pd.read_csv("D:\\4th Sem M.Sc Ds\\Major_project\\IKEA_SA_Furniture_Web_Scrapings_sss.csv")
ikea2 = pd.read_csv("D:\\4th Sem M.Sc Ds\\Major_project\\ikea.csv")

# Load Pinterest dataset
pinterest = pd.read_csv("D:\\4th Sem M.Sc Ds\\Major_project\\cleaned_decor.csv")

# Print shapes
print(f"IKEA Dataset 1: {ikea1.shape}")
print(f"IKEA Dataset 2: {ikea2.shape}")
print(f"Pinterest Dataset: {pinterest.shape}")


# In[ ]:





# In[ ]:





# In[2]:


ikea1.head()     # Check IKEA dataset 1


# In[ ]:





# In[3]:


pinterest.head() # Check Pinterest trends


# In[4]:


# cleaning data set


# In[5]:


# Check structure and null values
ikea1.info()
ikea2.info()

# Check for missing data
print("\nMissing in IKEA1:\n", ikea1.isnull().sum())
print("\nMissing in IKEA2:\n", ikea2.isnull().sum())


# In[6]:


# Check all column names in IKEA1
print("🔍 Columns in IKEA1:")
print(ikea1.columns.tolist())


# In[7]:


# Clean IKEA1

ikea1 = ikea1.drop_duplicates()
ikea1 = ikea1.dropna(subset=['name', 'price'])  # Use correct column names
ikea1['price'] = ikea1['price'].replace('[₹,]', '', regex=True).astype(float, errors='ignore')
ikea1['category'] = ikea1['category'].astype(str).str.strip().str.lower()
ikea1['name'] = ikea1['name'].astype(str).str.strip()


# In[8]:


pinterest.info()
pinterest.isnull().sum()


# In[9]:


print("📌 Pinterest Dataset Columns:")
print(pinterest.columns.tolist())


# In[10]:


['name', 'category', 'image_url', 'link']


# In[11]:


pinterest = pinterest.dropna(subset=['name', 'category'])
pinterest['category'] = pinterest['category'].str.strip().str.lower()
pinterest['name'] = pinterest['name'].str.strip()


# In[12]:


pinterest.columns.tolist()


# In[13]:


print("🔎 IKEA1 Columns:")
print(ikea1.columns.tolist())

print("\n🔎 IKEA2 Columns:")
print(ikea2.columns.tolist())

print("\n🔎 Pinterest Columns:")
print(pinterest.columns.tolist())


# In[14]:


#Cleaning Ikea dar


# In[15]:


ikea_cols_to_keep = ['name', 'category', 'price']
ikea1_cleaned = ikea1[ikea_cols_to_keep].dropna().drop_duplicates()
ikea1_cleaned = ikea1_cleaned.rename(columns={'name': 'product_name'})
ikea1_cleaned['brand'] = 'IKEA'

ikea2_cleaned = ikea2[ikea_cols_to_keep].dropna().drop_duplicates()
ikea2_cleaned = ikea2_cleaned.rename(columns={'name': 'product_name'})
ikea2_cleaned['brand'] = 'IKEA'


# In[16]:


# Ensure price column is numeric
pinterest['price'] = pd.to_numeric(pinterest['price'], errors='coerce')

pinterest_cleaned = pinterest[['name', 'category', 'price']].dropna()
pinterest_cleaned = pinterest_cleaned.rename(columns={'name': 'product_name'})
pinterest_cleaned['brand'] = 'Pinterest'


# In[17]:


combined_df = pd.concat([ikea1_cleaned, ikea2_cleaned, pinterest_cleaned], ignore_index=True)
combined_df = combined_df.dropna().drop_duplicates()

print("✅ Combined Dataset Shape:", combined_df.shape)
combined_df.head()


# In[20]:


get_ipython().system('pip uninstall seaborn -y')
get_ipython().system('pip install seaborn --upgrade')


# In[22]:


get_ipython().system('pip install matplotlib --upgrade')
get_ipython().system('pip install pandas --upgrade')
get_ipython().system('pip install scipy --upgrade')


# In[23]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[24]:


#Customer Preference Analysis


# In[26]:


#Basic Category Distribution


# In[25]:


import seaborn as sns
import matplotlib.pyplot as plt

# Ensure category column is lowercase and clean
combined_df['category'] = combined_df['category'].str.strip().str.lower()

# Count products per category
category_counts = combined_df['category'].value_counts().reset_index()
category_counts.columns = ['category', 'count']

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=category_counts.head(10), x='category', y='count', palette='viridis')
plt.title("🔍 Top 10 Popular Product Categories (Customer Preference Proxy)")
plt.xlabel("Category")
plt.ylabel("Number of Products")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[27]:


#Price Distribution Across Categories (Preference vs. Price)


# In[28]:


# Convert price to numeric if not already
combined_df['price'] = pd.to_numeric(combined_df['price'], errors='coerce')

# Drop rows where price is missing
price_data = combined_df.dropna(subset=['price', 'category'])

# Boxplot to see price spread
plt.figure(figsize=(14, 6))
sns.boxplot(data=price_data[price_data['category'].isin(price_data['category'].value_counts().head(8).index)],
            x='category', y='price', palette='coolwarm')
plt.title("💰 Price Distribution by Top Categories")
plt.xlabel("Category")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[29]:


#Add Rating Analysis (if Pinterest has 'rating')


# In[34]:


if 'rating' in pinterest.columns:
    pinterest['rating'] = pd.to_numeric(pinterest['rating'], errors='coerce')
    rating_data = pinterest.dropna(subset=['rating', 'category'])

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=rating_data[rating_data['category'].isin(rating_data['category'].value_counts().head(6).index)],
                x='category', y='rating', palette='magma')
    plt.title("⭐ Customer Ratings by Category (Pinterest)")
    plt.xlabel("Category")
    plt.ylabel("Rating")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# In[35]:


#Pinterest vs IKEA


# In[38]:


# Add source columns to each dataset BEFORE combining
ikea1['source'] = 'IKEA1'
ikea2['source'] = 'IKEA2'
pinterest['source'] = 'Pinterest'

# Combine all datasets into one
combined_df = pd.concat([ikea1, ikea2, pinterest], ignore_index=True)


# In[39]:


# Ensure 'source' and 'category' are clean
combined_df['source'] = combined_df['source'].str.strip().str.title()
combined_df['category'] = combined_df['category'].str.strip().str.lower()

# Remove rows with missing price or category
comparison_df = combined_df.dropna(subset=['price', 'category'])


# In[ ]:





# In[40]:


# Price Distribution Comparison


# In[41]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.histplot(data=comparison_df, x='price', hue='source', bins=50, kde=True, palette='Set2')
plt.title("💰 Price Distribution: Pinterest vs IKEA")
plt.xlabel("Price")
plt.ylabel("Count")
plt.xlim(0, comparison_df['price'].quantile(0.95))  # Limit to 95th percentile to avoid extreme outliers
plt.tight_layout()
plt.show()


# In[42]:


#Average Price by Category


# In[43]:


# Filter top categories
top_categories = comparison_df['category'].value_counts()[comparison_df['category'].value_counts() > 20].index
filtered_df = comparison_df[comparison_df['category'].isin(top_categories)]

# Group and visualize
avg_price = filtered_df.groupby(['category', 'source'])['price'].mean().reset_index()

plt.figure(figsize=(14, 6))
sns.barplot(data=avg_price, x='category', y='price', hue='source', palette='Paired')
plt.title("📊 Average Price by Category: Pinterest vs IKEA")
plt.ylabel("Average Price")
plt.xlabel("Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[44]:


plt.figure(figsize=(10, 5))
sns.countplot(data=comparison_df, x='category', hue='source', order=comparison_df['category'].value_counts().index[:10], palette='coolwarm')
plt.title("🔥 Top 10 Categories: Pinterest vs IKEA")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[45]:


# Convert 'price' column to numeric (remove errors like strings or missing)
combined_df['price'] = pd.to_numeric(combined_df['price'], errors='coerce')

# Drop rows where price or category is missing
pricing_df = combined_df.dropna(subset=['price', 'category', 'source'])

# Optional: remove extreme outliers (prices > 99th percentile)
pricing_df = pricing_df[pricing_df['price'] < pricing_df['price'].quantile(0.99)]


# In[49]:


#Visualize Price Distribution by Source


# In[46]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.boxplot(data=pricing_df, x='source', y='price', palette='Set3')
plt.title("💰 Price Distribution by Source (IKEA & Pinterest)")
plt.ylabel("Price (in local currency)")
plt.xlabel("Source")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# In[48]:


#Price Trends by Category and Source


# In[47]:


avg_price = pricing_df.groupby(['category', 'source'])['price'].mean().reset_index()

plt.figure(figsize=(14, 6))
sns.barplot(data=avg_price, x='category', y='price', hue='source', palette='Set2')
plt.title("📦 Average Price by Category and Source")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Average Price")
plt.xlabel("Category")
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[50]:


pip install scikit-learn pandas matplotlib


# In[51]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# Load IKEA and Pinterest (merged earlier as combined_df)
df_ml = combined_df[['name', 'category']].dropna()
df_ml = df_ml[df_ml['category'].str.len() > 2]  # basic filter

# Preprocess text
df_ml['name'] = df_ml['name'].str.lower().str.replace('[^a-zA-Z ]', '', regex=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(df_ml['name'], df_ml['category'], test_size=0.2, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

# Report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Try on new titles
sample_titles = ["Velvet armchair with metal legs", "Modern wall shelf", "Scandinavian coffee table"]
sample_vec = vectorizer.transform(sample_titles)
print("\nPredictions for sample titles:")
for title, cat in zip(sample_titles, model.predict(sample_vec)):
    print(f"📦 {title} ➜ {cat}")


# In[ ]:




