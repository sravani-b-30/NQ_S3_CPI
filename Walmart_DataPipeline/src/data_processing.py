import pandas as pd
from logger import logger
from datetime import datetime, timedelta

def preprocess_serp_data(merged_df):
    """
    Preprocesses SERP data by converting timestamps, extracting dates, and removing duplicates.
    """
    try:
        merged_df['scrapped_at'] = pd.to_datetime(merged_df['scrapped_at'], errors='coerce', format='mixed')
        merged_df.dropna(subset=['scrapped_at'], inplace=True)
        
        merged_df['date'] = merged_df['scrapped_at'].dt.date
        logger.info(f"Extracted date from scrapped_at column : {merged_df.shape}")
        logger.info(f"Datatype of date column : {merged_df['date'].dtype}")

        merged_df.sort_values(by=['walmart_id', 'date', 'scrapped_at'], inplace=True)
        
        merged_df = merged_df.groupby(['walmart_id', 'date']).tail(1)
        logger.info(f"Grouped by walmart_id and date : {merged_df.shape}")

        merged_df.drop(columns=['scrapped_at'], inplace=True)
        logger.info(f"Dropped scrapped_at column from the dataframe : {merged_df.columns}")

        merged_df.reset_index(drop=True, inplace=True)
        
        logger.info(f"Preprocessed SERP data: {merged_df.shape}")
        logger.info("Step 5 : Preprocessing SERP data completed successfully")
        return merged_df
    except Exception as e:
        logger.error(f"Error in preprocessing SERP data in data_processing.py : {e}")
        raise e

def perform_monthly_analysis(merged_df, days=60):
    """
    Performs a rolling monthly analysis on the dataset.
    """
    try:
        merged_df.rename(columns={'walmart_id': 'id'}, inplace=True)
        logger.info(f"Renamed walmart_id to id in SERP Dataset : {merged_df.columns}")

        merged_df['date'] = pd.to_datetime(merged_df['date'], format='mixed').dt.date

        
        dfs = []
        for i in range(days):
            logger.info(f"Processing ASIN data for day {i+1}/{days}")
            analysis_date = merged_df['date'].max() - timedelta(days=i)
            logger.info(f"Analysis Date: {analysis_date}")

            start_date = analysis_date - timedelta(days=30)
            logger.info(f"Start Date: {start_date}")
            
            last_30_days_df = merged_df[(merged_df['date'] <= analysis_date) & (merged_df['date'] > start_date)]
            logger.info(f"Filtered DataFrame for the last 30 days: {last_30_days_df.shape}")

            last_30_days_df = last_30_days_df.sort_values(by=['id', 'date'], ascending=[True, False])
            logger.info(f"Sample data of id and date after sorting : {last_30_days_df[['id', 'date']].head()}")
            
            unique_asins = last_30_days_df.groupby('id').agg({
                'product_title': 'first',
                'sale_price': 'first',
                'brand': 'first',
                'rank': 'first',
                'organic_search_rank': 'first',
                'sponsored_search_rank': 'first',
                'keyword': 'first',
                'keyword_id': 'first'
            }).reset_index()
            
            unique_asins['analysis_date'] = analysis_date
            dfs.append(unique_asins)
        
        final_df = pd.concat(dfs).reset_index(drop=True)
        final_df.rename(columns={'sale_price': 'price', 'analysis_date': 'date'}, inplace=True)
        
        logger.info(f"Monthly analysis completed: {final_df.shape}")
        logger.info("Step 6 : Monthly analysis of SERP Data completed successfully")

        return final_df
    except Exception as e:
        logger.error(f"Error in performing monthly analysis in data_processing.py: {e}")
        raise e

def clean_napqueen_data(nq_merged_df, merged_df):
    """
    Cleans and merges NapQueen data with SERP dataset.
    """
    try:
        nq_merged_df.rename(columns={'walmart_id': 'id'}, inplace=True)
        full_column_order = [
            'id', 'product_title', 'listingPrice', 'brand', 'rank', 
            'organic_search_rank', 'sponsored_search_rank', 'keyword', 'keyword_id', 'date', 'channel_type'
        ]
        for col in full_column_order:
            if col not in nq_merged_df.columns:
                nq_merged_df[col] = None
        
        nq_merged_df = nq_merged_df.reindex(columns=full_column_order)
        nq_merged_df.rename(columns={'listingPrice': 'price'}, inplace=True)
        logger.info(f"Reordered columns of NapQueen Dataset to match with SERP Dataset : {nq_merged_df.columns}")
        
        logger.info(f"No.of rows in SERP Dataset before excluding NapQueen Products : {merged_df.shape}")
        merged_df = merged_df[merged_df['brand'] != 'napqueen']
        logger.info(f"Filtered SERP Dataset after excluding NapQueen products : {merged_df.shape}")

        final_df = pd.concat([merged_df, nq_merged_df], ignore_index=True)
        
        logger.info(f"NapQueen data cleaned and appended with SERP Data: {final_df.shape}")
        logger.info("Step 9 : Cleaning NapQueen data and appending with SERP data completed successfully")
        return final_df
    except Exception as e:
        logger.error(f"Error in cleaning NapQueen data in data_processing.py: {e}")
        raise e 
