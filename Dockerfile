# Use an official Python runtime as a parent image
# We choose a slim version for smaller image size, based on Debian Buster
FROM python:3.9-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir: Prevents pip from storing cached files, reducing image size.
# --upgrade pip: Ensures pip is up-to-date.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the trained model directory into the container at /app
COPY model/ ./model/

# Copy the application directory into the container at /app
COPY app/ ./app/

# Expose port 5000 for the Flask application
EXPOSE 5000

# Command to run the application using Gunicorn
# Gunicorn is a production-ready WSGI HTTP Server for Python web applications.
# -b 0.0.0.0:5000: Binds Gunicorn to all network interfaces on port 5000.
# --workers 4: Specifies the number of worker processes (adjust based on CPU cores).
# app.app: Refers to the 'app' module (app.py) and the 'app' Flask instance within it.
CMD ["gunicorn", "-b", "0.0.0.0:5000", "--workers", "4", "app.app:app"]

