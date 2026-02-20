@echo off
start http://localhost:8501
streamlit run app.py --server.address localhost --server.port 8501 --server.headless true
pause
