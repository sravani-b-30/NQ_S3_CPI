import pandas as pd
import pg8000
import logger
from decimal import Decimal
import re
import nltk
from nltk.tokenize import word_tokenize
import ast

# def tokenize(text):
#     tokens = word_tokenize(text.lower())
#     return set(tokens)


# def tokenize_with_delimiters(text):
#     text = text.lower()
#     tokens = re.split(r'[,;.\s]', text)
#     return set(token for token in tokens if token)


# def extract_numeric_metric(text):
#     return set(re.findall(r'\d+\s?[a-zA-Z"]+', text.lower()))


# def extract_thickness(dimension_str):
#     match = re.search(r'\d+"Th', dimension_str)
#     if match:
#         return match.group()
#     return ''


# def tokenized_similarity(value1, value2):
#     if value1 is None or value2 is None:
#         return 0
#     tokens1 = tokenize_with_delimiters(str(value1))
#     tokens2 = tokenize_with_delimiters(str(value2))
#     numeric_metric1 = extract_numeric_metric(str(value1))
#     numeric_metric2 = extract_numeric_metric(str(value2))
#     intersection = tokens1.intersection(tokens2).union(numeric_metric1.intersection(numeric_metric2))
#     return len(intersection) / len(tokens1.union(tokens2))


# def jaccard_similarity(set1, set2):
#     intersection = set1.intersection(set2)
#     union = set1.union(set2)
#     return len(intersection) / len(union)


# def title_similarity(title1, title2):
#     # Tokenize the titles
#     title1_tokens = tokenize_with_delimiters(title1)
#     title2_tokens = tokenize_with_delimiters(title2)

#     # Calculate intersection and union of title tokens
#     intersection = title1_tokens.intersection(title2_tokens)
#     union = title1_tokens.union(title2_tokens)

#     # Calculate token similarity score
#     token_similarity_score = len(intersection) / len(union)

#     # Extract numeric metrics
#     numeric_metric1 = extract_numeric_metric(title1)
#     numeric_metric2 = extract_numeric_metric(title2)

#     # Calculate numeric metric match score
#     numeric_match_count = len(numeric_metric1.intersection(numeric_metric2))

#     # Final similarity score
#     similarity_score = (token_similarity_score + numeric_match_count) * 100

#     return similarity_score, title1_tokens, title2_tokens, intersection


# def description_similarity(desc1, desc2):
#     desc1_tokens = tokenize_with_delimiters(desc1)
#     desc2_tokens = tokenize_with_delimiters(desc2)
#     intersection = desc1_tokens.intersection(desc2_tokens)
#     union = desc1_tokens.union(desc2_tokens)
#     similarity_score = len(intersection) / len(union) * 100
#     return similarity_score, desc1_tokens, desc2_tokens, intersection


# def parse_dict_str(dict_str):
#     try:
#         return ast.literal_eval(dict_str)
#     except ValueError:
#         return {}


# def merge_dicts(dict1, dict2):
#     merged = dict1.copy()
#     merged.update(dict2)
#     return merged


# def convert_weight_to_kg(weight_str):
#     weight_str = weight_str.lower()
#     match = re.search(r'(\d+\.?\d*)\s*(pounds?|lbs?|kg)', weight_str)
#     if match:
#         value, unit = match.groups()
#         value = float(value)
#         if 'pound' in unit or 'lb' in unit:
#             value *= 0.453592
#         return value
#     return None


# def parse_weight(weight_str):
#     weight_kg = convert_weight_to_kg(weight_str)
#     return weight_kg


# def parse_dimensions(dimension_str):
#     matches = re.findall(r'(\d+\.?\d*)\s*"?([a-zA-Z]+)"?', dimension_str)
#     if matches:
#         return {unit: float(value) for value, unit in matches}
#     return {}


# def compare_weights(weight1, weight2):
#     weight_kg1 = parse_weight(weight1)
#     weight_kg2 = parse_weight(weight2)
#     if weight_kg1 is not None and weight_kg2 is not None:
#         return 1 if abs(weight_kg1 - weight_kg2) < 1e-2 else 0
#     return 0


# def compare_dimensions(dim1, dim2):
#     dim1_parsed = parse_dimensions(dim1)
#     dim2_parsed = parse_dimensions(dim2)
#     if not dim1_parsed or not dim2_parsed:
#         return 0
#     matching_keys = set(dim1_parsed.keys()).intersection(set(dim2_parsed.keys()))
#     matching_score = sum(1 for key in matching_keys if abs(dim1_parsed[key] - dim2_parsed[key]) < 1e-2)
#     total_keys = len(set(dim1_parsed.keys()).union(set(dim2_parsed.keys())))
#     return matching_score / total_keys


# def calculate_similarity(details1, details2, title1, title2, desc1, desc2):
#     score = 0
#     total_keys = len(details1.keys())
#     details_comparison = []
#     for key in details1.keys():
#         if key in details2:
#             value1 = str(details1[key])
#             value2 = str(details2[key])
#             if 'weight' in key.lower():
#                 match_score = compare_weights(value1, value2)
#                 details_comparison.append(f"{key}: {value1} vs {value2} -> Match: {match_score}")
#                 score += match_score
#             elif 'dimension' in key.lower() or key.lower() == 'product dimensions':
#                 match_score = compare_dimensions(value1, value2)
#                 details_comparison.append(f"{key}: {value1} vs {value2} -> Match Score: {match_score}")
#                 score += match_score
#             else:
#                 match_score = tokenized_similarity(value1, value2)
#                 details_comparison.append(f"{key}: {value1} vs {value2} -> Match Score: {match_score}")
#                 score += match_score
#     if total_keys > 0:
#         details_score = (score / total_keys) * 100
#     else:
#         details_score = 0
#     title_score, title1_tokens, title2_tokens, title_intersection = title_similarity(title1, title2)
#     title_comparison = f"Title Tokens (Target): {title1_tokens}\nTitle Tokens (Competitor): {title2_tokens}\nCommon Tokens: {title_intersection}\nScore: {title_score}"
#     desc_score, desc1_tokens, desc2_tokens, desc_intersection = description_similarity(desc1, desc2)
#     desc_comparison = f"Description Tokens (Target): {desc1_tokens}\nDescription Tokens (Competitor): {desc2_tokens}\nCommon Tokens: {desc_intersection}\nScore: {desc_score}"

#     return details_score, title_score, desc_score, details_comparison, title_comparison, desc_comparison


# def calculate_weighted_score(details_score, title_score, desc_score):
#     weighted_score = 0.5 * details_score + 0.4 * title_score + 0.1 * desc_score
#     return weighted_score


# def calculate_cpi_score(target_price, competitor_prices):
#     percentile = 100 * (competitor_prices < target_price).mean()
#     cpi_score = 10 - (percentile / 10)
#     return cpi_score

# def extract_brand_from_title(title):
#     if pd.isna(title) or not title:
#         return 'unknown'
#     return title.split()[0].lower()


# def extract_style(title):
#     title = str(title)
#     style_pattern = r"\b(\d+)\s*(inches?|in|inch|\"|''|'\s*'\s*)\b"
#     style_match = re.search(style_pattern, title.lower())

#     if style_match:
#         number = style_match.group(1)
#         return f"{number} Inch"

#     style_pattern_with_quote = r"\b(\d+)\s*(''{1,2})"
#     style_match = re.search(style_pattern_with_quote, title.lower())

#     if style_match:
#         number = style_match.group(1)
#         return f"{number} Inch"
#     return None


# def extract_size(title):
#     title = str(title)
#     size_patterns = {
#         'Twin XL': r'\btwin[-\s]xl\b',
#         'Queen': r'\bqueen\b',
#         'Full': r'\b(full|double)\b',
#         'Twin': r'\btwin\b',
#         'King': r'\bking\b'
#     }

#     title_lower = title.lower()

#     for size, pattern in size_patterns.items():
#         if re.search(pattern, title_lower):
#             return size

#     return None

# merged_data_df = pd.read_csv("merged_data_2024-12-25.csv")

# merged_data_df['ASIN'] = merged_data_df['ASIN'].str.upper()

# missing_brand_mask = merged_data_df['brand'].isna() | (merged_data_df['brand'] == "")
# merged_data_df.loc[missing_brand_mask, 'brand'] = merged_data_df.loc[missing_brand_mask, 'product_title'].apply(extract_brand_from_title)

# merged_data_df['price'] = pd.to_numeric(merged_data_df['price'], errors='coerce')

# merged_data_df['Product Details'] = merged_data_df['Product Details'].apply(parse_dict_str)
# merged_data_df['Glance Icon Details'] = merged_data_df['Glance Icon Details'].apply(parse_dict_str)
# merged_data_df['Option'] = merged_data_df['Option'].apply(parse_dict_str)
# merged_data_df['Drop Down'] = merged_data_df['Drop Down'].apply(parse_dict_str)

# merged_data_df['Style'] = merged_data_df['product_title'].apply(extract_style)
# merged_data_df['Size'] = merged_data_df['product_title'].apply(extract_size)

# def update_product_details(row):
#     details = row['Product Details']
#     details['Style'] = row['Style']
#     details['Size'] = row['Size']
#     return details

# merged_data_df['Product Details'] = merged_data_df.apply(update_product_details, axis=1)

# def extract_dimensions(details):
#     if isinstance(details, dict):
#         return details.get('Product Dimensions', None)
#     return None

# merged_data_df['Product Dimensions'] = merged_data_df['Product Details'].apply(extract_dimensions)

# # reference_df = pd.read_csv('List_CPI/product_dimension_size_style_reference.csv')

# # merged_data_df = merged_data_df.merge(reference_df, on='Product Dimensions', how='left', suffixes=('', '_ref'))

# # # Fill missing values in 'Size' and 'Style' columns with the values from the reference DataFrame
# # merged_data_df['Size'] = merged_data_df['Size'].fillna(merged_data_df['Size_ref'])
# # merged_data_df['Style'] = merged_data_df['Style'].fillna(merged_data_df['Style_ref'])

# # merged_data_df['Product Details'] = merged_data_df.apply(update_product_details, axis=1)

# merged_data_df.to_csv("merged_data_after_processing.csv", index=False)

df1 = pd.read_csv("merged_data_2024-12-25.csv", on_bad_lines='skip')
df2 = pd.read_csv("merged_data_2024-12-27.csv", on_bad_lines='skip')

df = pd.concat([df1,df2], ignore_index=True)
print(len(df['ASIN']))
print(df.info())
df.to_csv("merged_data_2024-12-27_fixed.csv", index=False)