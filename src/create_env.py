import os

os. system('conda config --add channels conda-forge')
os. system('conda create -n emotion-detection python=3.5 scipy pandas pandasql sqlite scikit-learn matplotlib opencv jupyter psqlodbc psycopg2 anaconda --yes')
os. system('conda activate emotion-detection')
