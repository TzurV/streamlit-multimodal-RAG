@echo off

REM Build Docker image
docker build -t streamlit-app .

if %errorlevel% equ 0 (
  echo Image built successfully!
  echo Connect to http://localhost:8001/

  REM Run container with auto-remove
  docker run -p 8001:8001 --rm streamlit-app

  echo Container stopped and removed.
) else (
  echo Error building Docker image
  exit /b %errorlevel%
)