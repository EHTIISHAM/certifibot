# 1. Use an official lightweight Python image
FROM python:3.9-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy dependencies first (Caching)
COPY requirements.txt .

# 4. Install dependencies
# We add build-essential for some python packages
RUN apt-get update && apt-get install -y build-essential \
    && pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your app
COPY . .

# 6. Create the DB directory
RUN mkdir -p /app/cert_db

# 7. Expose the port
EXPOSE 8000

# 8. Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]