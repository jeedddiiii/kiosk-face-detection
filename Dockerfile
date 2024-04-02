# Use the official Python 3.8.8 image as the base image
FROM python:3.8.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory
COPY . .

# Expose the port that the Flask app will run on
EXPOSE 5000

# Run the Flask application
CMD ["python", "app.py"]