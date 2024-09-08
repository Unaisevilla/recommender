# Use the official lightweight Python image.
FROM python:3.9-slim

# Set the working directory in the container.
WORKDIR /app

# Copy the dependencies file to the working directory.
COPY requirements.txt .

# Install any dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local app directory to the working directory.
COPY app/ app/

# Copy the content of the local templates directory to the working directory.
COPY templates/ templates/

# Copy the content of the local static directory to the working directory.
COPY static/ static/

# Copy the content of the local data directory to the working directory.
COPY data/ data/

# Set the PYTHONPATH environment variable
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Command to run the application.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]