@echo on

cd C:\Users\plang\Desktop\2. Leuven\Thesis\1. Code

REM Activate the CharacterBERT conda environment
call conda activate character-bert

REM Run cldr_clner_embeddings.py with argument 1
python cldr_clner_embeddings.py 1

REM Run cldr_clner_embeddings.py with argument 2
python cldr_clner_embeddings.py 2

REM Run cldr_clner_embeddings.py with argument 3
python cldr_clner_embeddings.py 3

REM Run cldr_clner_embeddings.py with argument 4
python cldr_clner_embeddings.py 4

REM Run cldr_clner_embeddings.py with argument 5
python cldr_clner_embeddings.py 5

REM Run cldr_clner_embeddings.py with argument 6
python cldr_clner_embeddings.py 6

REM Run cldr_clner_embeddings.py with argument 7
python cldr_clner_embeddings.py 7

REM Run cldr_clner_embeddings.py with argument 8
python cldr_clner_embeddings.py 8

REM Run cldr_clner_embeddings.py with argument 9
python cldr_clner_embeddings.py 9

REM Run cldr_clner_embeddings.py with argument 10
python cldr_clner_embeddings.py 10

REM Deactivate the conda environment
call conda deactivate

REM Prompt the user to type "exit" to close the window
set /p close=Type "exit" to close the window: 
if "%close%"=="exit" exit