# Runs the exact ILP optimizer for any budget:
#   docker build -t cooling-optimizer .
#   docker run --rm cooling-optimizer --budget "50 lakh"
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT ["python", "optimize.py"]
CMD ["--budget", "10 cr"]
