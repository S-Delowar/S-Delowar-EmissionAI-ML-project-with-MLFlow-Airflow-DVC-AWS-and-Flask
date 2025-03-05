# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt from flask/ to /app
COPY ./flask/requirements.txt . 

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app.py from flask/ to /app
COPY ./flask/app.py . 

# Environment and ports
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Run the app
CMD ["python", "app.py"]


# docker build -f flask/Dockerfile -t emission-ai-flask-app .
# docker run --env-file .env -p 8001:8000 emission-ai-flask-app
