# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any Python dependencies required by our app
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of our app's code into the container
COPY . /app

# Expose a different port to avoid conflict with the first app
EXPOSE 8503

# Command to run the second app (update script name and port if needed)
CMD ["streamlit", "run", "walmart_nq.py", "--server.port=8503", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]

