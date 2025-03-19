import pandas as pd
import boto3
from logger import logger
from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME, S3_FOLDER
from io import StringIO
import io

# Initialize S3 client
s3_client = boto3.client(
    's3'
    # aws_access_key_id=AWS_ACCESS_KEY_ID,
    # aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

def fetch_product_details_from_s3(bucket_name, s3_folder, file_name):
    """Fetch existing product details file from S3 as a Pandas DataFrame with a fallback for missing files."""
    s3_key = f"{s3_folder}/{file_name}"
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        df = pd.read_csv(response["Body"])
        logger.info(f"Successfully fetched existing product details from S3: {file_name}")
        return df
    except s3_client.exceptions.NoSuchKey:
        logger.warning(f"File {s3_key} not found in S3. Creating a new empty product details file.")
        return pd.DataFrame(columns=["ID", "URL", "SKU", "GTIN", "Price", "Title", "Rating", "Rating Count",
                                     "Seller ID", "Seller Name", "Currency", "Description", "Out of Stock",
                                     "Specifications"])
    except Exception as e:
        logger.error(f"Error fetching file from S3: {e}. Creating an empty file as fallback.")
        return pd.DataFrame(columns=["ID", "URL", "SKU", "GTIN", "Price", "Title", "Rating", "Rating Count",
                                     "Seller ID", "Seller Name", "Currency", "Description", "Out of Stock",
                                     "Specifications"])

def upload_file_to_s3(df, bucket_name, s3_folder, file_name):
    """Upload the updated product details file to S3."""
    try:
            s3_key = f"{s3_folder}/{file_name}" if s3_folder else file_name  # Ensure correct path

            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            
            s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=csv_buffer.getvalue()
            )
            logger.info(f"Successfully uploaded file to S3: {s3_key}")
    except Exception as e:
        logger.error(f"Error uploading file to S3: {e}")