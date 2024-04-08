FROM python:3.9.19-alpine3.19

WORKDIR /app

RUN apk add --no-cache g++ gcc libxslt-dev

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]