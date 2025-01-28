import pandas as pd
from datetime import datetime, timedelta
import ast
import re

end_date = datetime.now().date()
start_date = end_date - timedelta(days=3)
today = datetime.now()
cut_off_date = today.date() - timedelta(days=31)
print(end_date)
print(start_date)
print(today)
print(cut_off_date)

# price_data = pd.read_csv("Pipeline\sp_api_24_23_dec.csv")

# price_data['date'] = pd.to_datetime(price_data['date']).dt.date
# unique_dates = price_data['date'].unique()
# print(f"Printing unique dates")
# print(sorted(unique_dates))

# date_counts = price_data['date'].value_counts()

# # Display the counts
# print("Frequency of dates in the dataframe:")
# print(date_counts)

# filtered_data = price_data[price_data['date'] == pd.to_datetime('2024-12-23').date()]

# # Display the filtered dataframe
# print(f"Filtered data for 23rd December 2024:")
# print(filtered_data.head())

# # Optionally, check the number of rows in the filtered dataframe
# print(f"Number of rows for 23rd December 2024: {len(filtered_data)}")

"""Merge serp and scrapped file"""

# df = pd.read_csv("C:/Users/bande/Recommended CPI/serp_sp_api_data_15th.csv", on_bad_lines='skip')
# df_scrapped_info = pd.read_csv("C:/Users/bande/Downloads/NAPQUEEN (8).csv")

# df_scrapped_info = df_scrapped_info[df_scrapped_info['Option'] != '{}']

#     # Rename 'asin' to 'ASIN' in df to match df_scrapped_info column
# df.rename(columns={'asin': 'ASIN'}, inplace=True)

# # Merge the DataFrames on the 'ASIN' column
# merged_df = pd.merge(df, df_scrapped_info, on='ASIN', how='left')

# merged_df.to_csv("merged_data_11-15_Jan.csv", index=False)