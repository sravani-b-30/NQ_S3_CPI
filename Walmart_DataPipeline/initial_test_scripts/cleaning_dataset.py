import pandas as pd
import ast
import numpy as np

# Load the CSV file
file_path = 'PRODUCT_DETAILS.csv'
df = pd.read_csv(file_path, dtype=str)

# # Iterate through rows and keep only those where the first column starts with a digit
# def is_valid_row(row):
#     first_column_value = str(row.iloc[0]).strip()  # Get the first column value
#     return first_column_value.isdigit()  # Check if it starts with a numeric digit

# # Apply row filtering using iteration
# df = df[df.apply(is_valid_row, axis=1)]

# Define default values for cleaning
default_values = {
    "SKU": {},
    "product_name": None,
    "Description": None,
    "Price": None,
    "Title": None,
    "Rating": None,
    "Rating Count": None,
    "Seller ID": {},
    "Seller Name": None,
    "Currency": None,
    "Out of Stock": None,
    "Specifications": {}
}

# Define a function to clean error messages
def clean_field(value, default):
    if isinstance(value, str) and "Error while parsing" in value:
        return default  # Replace with the default value
    return value

# Apply cleaning function to relevant columns
for field, default in default_values.items():
    if field in df.columns:  # Only clean columns that exist in the data
        df[field] = df[field].apply(lambda x: clean_field(x, default))

# # Clean and reformat the Specifications column
# def parse_specifications(spec_str):
#     try:
#         # Convert JSON-like strings to Python dicts
#         return ast.literal_eval(spec_str)
#     except (ValueError, SyntaxError):
#         return default_values['Specifications']  # Replace invalid specs with the default value

# if 'Specifications' in df.columns:
#     df['Specifications'] = df['Specifications'].apply(parse_specifications)

# def row_starts_with_number(row):
#         first_char = str(row).strip()[0] if str(row).strip() else ''
#         return first_char.isdigit()

# # 🚀 Apply function to filter rows
# df = df[df.apply(lambda row: row_starts_with_number(row.to_string(index=False)), axis=1)]


# Save the cleaned data to a new file
cleaned_file_path = 'cleaned_dataset_sample.csv'
df.to_csv(cleaned_file_path, index=False)

print(f"Cleaned data saved to {cleaned_file_path}")
