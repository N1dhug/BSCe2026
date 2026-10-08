@echo off
setlocal
pushd "%~dp0"
python -m src.exe.run_model.bm_1_f4 %*
set "F4_EXIT=%ERRORLEVEL%"
popd
exit /b %F4_EXIT%
