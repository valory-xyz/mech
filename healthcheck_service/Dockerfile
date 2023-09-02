# Use the official Python 3.10 image as the base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the Python script into the container
COPY . /app/

RUN pip install -r requirements.txt

# Expose the port on which the health check server will run
EXPOSE 8080

# Run the Python script when the container starts
CMD ["python", "healthcheck.py"]
