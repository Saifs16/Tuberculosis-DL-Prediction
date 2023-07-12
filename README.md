# Deep-Learning TB Prediction

## Commands

1. Create a virtual environment

```
python -m venv .env
```

2. Activate the virtual environment

```
source .env/bin/activate
```

3. Install the requirements

```
pip install -r requirements.txt --no-cache-dir
```

4. Run the app

```
python app.py
```

---- With WSGI (Production)


- To start the server: `gunicorn -w 1 -b 0.0.0.0:5000 --timeout 600 wsgi:app`

- To start the server in background: `gunicorn -w 1 -b 0.0.0.0:5000 --timeout 600 wsgi:app --daemon`

- To stop the guincorn server: `ps ax|grep gunicorn` & `kill -9 <pid number>`