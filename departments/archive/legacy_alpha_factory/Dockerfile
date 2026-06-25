# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install basic Linux build tools (required for some Python math libraries)
RUN apt-get update && apt-get install -y build-essential gcc

# Copy your requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files (code, models, database) into the container
COPY . .

# Expose port 8501 for the Streamlit Scoreboard
EXPOSE 8501

# By default, when the container starts, it will run Streamlit
# We will use a separate command later to trigger the cron job
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]