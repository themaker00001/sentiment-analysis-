#Base image

FROM python:3.10-slim

#setting work dir
WORKDIR /app

# installing depentic 

COPY app/requirement.txt .
RUN pip install -r requirement.txt

#copy the app 

COPY app/ .

# exposing the 8000 port no 

EXPOSE 8000

# command for running afile 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]