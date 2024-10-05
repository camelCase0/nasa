FROM python:3.9-slim
WORKDIR /code
COPY ./requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt && pip cache purge
COPY . .
EXPOSE 80