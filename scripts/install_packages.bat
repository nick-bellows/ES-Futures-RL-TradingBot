@echo off
echo Installing packages...
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements_simple.txt
echo.
echo Installation complete!
pause
