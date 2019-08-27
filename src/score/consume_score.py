import json
import argparse
import score

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--DATA_FILE')
ARGS = PARSER.parse_args()

with open(ARGS.DATA_FILE) as json_file:
    MYDATA = json.load(json_file)

score.init()

RESULT = score.run(MYDATA)

print(RESULT)
