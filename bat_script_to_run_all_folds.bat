@echo on

cd C:\Users\plang\Desktop\2. Leuven\Thesis\1. Code

REM Activate the NewThesis conda environment
call conda activate NewThesis

REM Run main.py with argument 0
python main.py CLDR_CLNER 0

REM Run main.py with argument 1
python main.py CLDR_CLNER 1

REM Run main.py with argument 2
python main.py CLDR_CLNER 2

REM Run main.py with argument 3
python main.py CLDR_CLNER 3

REM Run main.py with argument 4
python main.py CLDR_CLNER 4

REM Run main.py with argument 5
python main.py CLDR_CLNER 5

REM Run main.py with argument 6
python main.py CLDR_CLNER 6

REM Run main.py with argument 7
python main.py CLDR_CLNER 7

REM Run main.py with argument 8
python main.py CLDR_CLNER 8

REM Run main.py with argument 9
python main.py CLDR_CLNER 9

REM Run avg_results.py to calculate average evaluation scores
python avg_results.py CLDR_CLNER

REM Deactivate the conda environment
call conda deactivate

REM Prompt the user to type "exit" to close the window
set /p close=Type "exit" to close the window: 
if "%close%"=="exit" exit