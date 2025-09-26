# fuzzprocess
Deep Learning Approach to Find Nearest String Match (Top K)

## Installation
pip install -U git+https://github.com/leoliu0/fuzztools

## Usage
Two steps, first identify sensible pairs of matching by 
```
fuzzprocess file1.csv file2.csv -o output.tsv
```
use -b flag to use best model, but for memory constraint device, use normal model is good, also faster. See --help for other options

Second step to use post process to find real good pairs
```
fuzzpost output.tsv
```
which requires a gemini API key, you need to export it in shell, i.e. export GEMINI_API_KEY=xxxxxxxxxxxx
