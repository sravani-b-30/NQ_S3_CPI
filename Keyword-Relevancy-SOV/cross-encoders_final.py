# clean_relevancy_pipeline.py

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import boto3
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------
# 1. Load Cross-Encoder Model
# ---------------------------

model_name = "cross-encoder/stsb-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# ---------------------------
# 2. Relevancy Scoring using Cross-Encoder
# ---------------------------

def cross_encoder_score(keyword, product_text):
    if not isinstance(product_text, str):
        return np.nan

    inputs = tokenizer(
        keyword,
        product_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.logits[0].item()

    return score

# ---------------------------
# 3. Load ASIN + Product Metadata and Tracked Keywords
# ---------------------------

product_df = pd.read_csv("Keyword-Relevancy-SOV/Files/CRAZY_CUPS_PRODUCT_DETAILS.csv", encoding='cp1252', on_bad_lines='skip')
product_df.rename(columns={'ASIN' : 'asin', 'Title' : 'title'}, inplace=True)

tracked_keywords = pd.read_csv("Keyword-Relevancy-SOV/Files/crazy_cups_tracked_keywords.csv", on_bad_lines='skip')
tracked_keywords = tracked_keywords[['keyword']]

search_query_df = pd.read_csv('Keyword-Relevancy-SOV/Files/SQP_DATA_CRAZY_CUPS.csv', on_bad_lines='skip')
search_query_df = search_query_df[['asin', 'searchQuery', 'searchQueryScore', 'startDate', 'endDate']]

# ---------------------------
# 4. Cross Join ASINs × Keywords
# ---------------------------

product_df['key'] = 1
tracked_keywords['key'] = 1
pairs_df = pd.merge(product_df, tracked_keywords, on='key').drop('key', axis=1)

pairs_df['combined_text'] = pairs_df['title'] + ' ' + pairs_df['Product Details']

# ---------------------------
# 5. Score Each Pair in Parallel
# ---------------------------

inputs = list(zip(pairs_df['keyword'], pairs_df['combined_text']))
with ThreadPoolExecutor() as executor:
    scores = list(tqdm(executor.map(lambda x: cross_encoder_score(*x), inputs), total=len(inputs), desc="Scoring Pairs"))

pairs_df['relevancy_score'] = scores

# ---------------------------
# 6. Merge Search Query Rank and Normalize
# ---------------------------

search_query_df['startDate'] = pd.to_datetime(search_query_df['startDate'])
search_query_df['endDate'] = pd.to_datetime(search_query_df['endDate'])

logger.info("\nSearch Query Data Loaded:\n")
logger.info(search_query_df.columns)
logger.info(search_query_df.head())
logger.info(search_query_df.info())

week_bins = search_query_df[['startDate', 'endDate']].drop_duplicates().reset_index(drop=True)
week_bins['startDate'] = pd.to_datetime(week_bins['startDate'])
week_bins['endDate'] = pd.to_datetime(week_bins['endDate'])

def find_week_bin(row, bins):
    for _, bin_row in bins.iterrows():
        if bin_row['startDate'] <= row['date'] <= bin_row['endDate']:
            return bin_row['startDate'], bin_row['endDate']
    return pd.NaT, pd.NaT

# Apply to each row
pairs_df['date'] = pd.to_datetime(pairs_df['date'])
pairs_df[['startDate', 'endDate']] = pairs_df.apply(lambda row: pd.Series(find_week_bin(row, week_bins)), axis=1)

# Drop rows that didn't fall into any week
pairs_df = pairs_df.dropna(subset=['startDate'])
pairs_df['startDate'] = pd.to_datetime(pairs_df['startDate'])
pairs_df['endDate'] = pd.to_datetime(pairs_df['endDate'])

weekly_rel_df = pairs_df.groupby(['asin', 'keyword', 'startDate', 'endDate']).agg({
    'relevancy_score': 'mean'
}).reset_index()
logger.info("\nWeekly Relevancy Data:\n")
logger.info(weekly_rel_df.columns)
logger.info(weekly_rel_df.head())
logger.info(weekly_rel_df.info())


combined_df = pd.merge(weekly_rel_df, search_query_df,
                       left_on=['asin', 'keyword', 'startDate', 'endDate'],
                       right_on=['asin', 'searchQuery', 'startDate', 'endDate'],
                       how='left')

min_rel = combined_df['relevancy_score'].min()
max_rel = combined_df['relevancy_score'].max()
combined_df['rel_score_norm'] = (combined_df['relevancy_score'] - min_rel) / (max_rel - min_rel)

combined_df['sqp_score_norm'] = 1 - ((combined_df['searchQueryScore'] - 1) / 99)

# ---------------------------
# 7. Classify Relevancy Buckets
# ---------------------------

def classify(row):
    if row['rel_score_norm'] >= 0.67 and row['sqp_score_norm'] >= 0.67:
        return 'Highly Relevant'
    elif row['rel_score_norm'] >= 0.33 and row['sqp_score_norm'] >= 0.33:
        return 'Moderately Relevant'
    else:
        return 'Less Relevant'

combined_df['relevancy_bucket'] = combined_df.apply(classify, axis=1)

# ---------------------------
# 8. Final Output
# ---------------------------

logger.info("\nRelevancy Buckets for Tracked Keywords (Crazy Cups):\n")
logger.info(combined_df[['asin', 'keyword', 'relevancy_score', 'searchQueryScore', 'rel_score_norm', 'sqp_score_norm', 'relevancy_bucket']])

# Export if needed
combined_df.to_csv("crazy_cups_keyword_relevancy.csv", index=False)

client = boto3.client('s3')
bucket_name = 'anarix-cpi'

file_name = 'crazy_cups_keyword_relevancy.csv'
s3_file_path = 'ANALYSIS/' + file_name

# Upload to S3
try:
    client.upload_file(combined_df, bucket_name, s3_file_path)
    logger.info(f"File {file_name} uploaded to S3 bucket {bucket_name}.")
except Exception as e:
    logger.error(f"Error uploading file to S3: {e}")