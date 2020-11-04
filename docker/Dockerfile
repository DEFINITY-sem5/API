FROM python:3.8

COPY . /app
COPY . .

RUN pip install -r req.txt

ENTRYPOINT [ "python" ]
CMD ["app.py"]