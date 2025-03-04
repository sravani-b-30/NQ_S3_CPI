from pymongo import MongoClient
import pandas as pd

# Replace with your actual MongoDB connection string
MONGO_URI = "mongodb+srv://read_only_prod_user:L4iWfhbuJXncNDeJr7sf3y7tGQjtkekR@anarix-serverless.f02zx6m.mongodb.net/?retryWrites=true&w=majority&appName=Anarix-Serverless"
DATABASE_NAME = "anarix_prod"  # Replace with actual database name

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]

# Collections
collection1 = db["walmartmarketplaceaccounts"]  # Replace with actual collection name
collection2 = db["serpkeywords"]  # Replace with actual collection name

accountId_to_brand = {}

# Fetch all documents from first collection
first_collection_data = collection1.find({})
for doc in first_collection_data:
    accountId = str(doc["accountId"])  # Convert ObjectId to string
    partnerDisplayName = doc["partnerDisplayName"]
    accountId_to_brand[accountId] = partnerDisplayName

# Step 2: Fetch all documents from second collection and apply mapping
second_collection_data = list(collection2.find({}))

# Convert second collection data to DataFrame
df = pd.DataFrame(second_collection_data)

# Extract accountId values and convert them to strings
df["accountId"] = df["accountId"].astype(str)

# Step 3: Create a 'brand' column using mapping
df["brand"] = df["accountId"].map(accountId_to_brand)

# Step 4: Function to filter data by brand
def filter_by_brand(brand_name):
    filtered_df = df[df["brand"] == brand_name]
    return filtered_df

# Example usage
brand_name_input = "NapQueen"  # Change this as per your need
filtered_data = filter_by_brand(brand_name_input)

# Print and save output
print(filtered_data)  # Print in console
filtered_data.to_csv("filtered_data.csv", index=False)  # Save to CSV