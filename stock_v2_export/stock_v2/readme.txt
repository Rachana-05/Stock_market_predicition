TO RUN UI AND MLFLOW :

ON WINDOWS POWER SHELL - 

-first terminal
    python -m uvicorn api.main:app --reload --port 8000

-2nd terminal:
    python -m mlflow ui --backend-store-uri "file:///C:/Users/racha/Downloads/stock_v2_export (2)/stock_v2_export/stock_v2/mlruns" --port 5000
C:\Users\racha\Downloads\stock_v2_export (2)\stock_v2_export\stock_v2\mlruns
- NOW OPEN THE INDEX FILE IN ANY BROWSER

TO PERFORM PYUNIT AND PYLINT TEST :

cd C:\Users\ADDRESS\stock_v2_export\stock_v2
pip install pytest pylint
python run_tests.py


