@echo off
echo Copying DataBento CSV files to project...
echo.
echo Please ensure your DataBento CSV files are in the source directory.
echo Source files expected:
echo   - ESU4_20240901_20240907.csv
echo   - ESH3_20221216_20230317.csv
echo   - ESU4_20240614_20240913.csv
echo   - ESZ3_20230915_20231215.csv
echo   - ESH4_20231215_20240315.csv
echo   - ESM3_20230317_20230616.csv
echo   - ESM4_20240315_20240614.csv
echo   - ESU3_20230616_20230915.csv
echo.
set /p source="Enter the path to your DataBento CSV files: "
xcopy "%source%\*.csv" "D:\QC_TradingBot_v3\data\databento\" /Y
echo.
echo Files copied to D:\QC_TradingBot_v3\data\databento\
pause
