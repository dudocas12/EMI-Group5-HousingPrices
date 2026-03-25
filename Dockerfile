# Start from the official Apache Airflow image
FROM apache/airflow:3.1.3

# Copy your local requirements file into the container
COPY requirements.txt /

# Install the dependencies Group 5 needs for the ML pipeline
RUN pip install --no-cache-dir -r /requirements.txt