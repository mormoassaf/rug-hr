### Environment variables
```
APP_NAME = BERTaaS
VERSION = 1.0.0
DESCRIPTION = BERT as a Service
WHITELIST = ["*"]
```

Build docker command
```shell
docker-compose up -d --build 
```

Run with a python virtual environment
```shell
rm -r env 
python3.9 -m venv env  
source env/bin/activate 
python3.9 -m pip install --upgrade pip
python3.9 -m pip install --no-cache-dir --upgrade -r requirements.txt
```
Exit with `$ deactivate`

Run the API
```
python3.9 -m uvicorn app.main:app --port 8080
```

Packaging lambda function
```
cd env/lib/python3.9/site-packages
zip -r9 ../../../../function.zip .
cd ../../../../
zip -g ./function.zip -r app
zip -g ./function.zip .env
```