# Use official lightweight Python image
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Copy necessary files
COPY requirements-docker.txt ./
COPY app/ ./app/
COPY data/processed/tagged_description.txt ./data/processed/tagged_description.txt
COPY data/processed/books_with_emotions.csv ./data/processed/books_with_emotions.csv
COPY data/assets/cover-not-found.jpg ./data/assets/cover-not-found.jpg

# Install dependencies
RUN pip install --no-cache-dir -r requirements-docker.txt

# Expose Gradio default port
EXPOSE 7860

# Command to run the app
CMD ["python", "app/gradio_dashboard.py"]
