@echo off
echo Testing Environment with Virtual Environment Active
echo ===================================================

if not exist "venv" (
    echo ERROR: Virtual environment not found!
    echo Please run fix_environment.bat first.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Running verification tests...
python verify_setup.py

pause