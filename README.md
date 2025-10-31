
Test 0: replay attack (similarity score = 1.0)

python behavio.py eval  --models-dir ./models --csv session_mickey_0.csv --username "Mickey Mouse" --contract "795131459" --out-csv ./scores_mickey_0.csv


Test 1: usual typing speed (high similarity score = 0.9128005213572183)

python behavio.py eval  --models-dir ./models --csv session_mickey_1.csv --username "Mickey Mouse" --contract "795131459" --out-csv ./scores_mickey_1.csv


Test 2: unsually slow typing speed (low similarity score = 0.09811024818851996)

python behavio.py eval  --models-dir ./models --csv session_mickey_2.csv --username "Mickey Mouse" --contract "795131459" --out-csv ./scores_mickey_2.csv 
