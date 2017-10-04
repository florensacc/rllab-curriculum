import json
import pprint
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, default=None)
    parser.add_argument('-a', '--all', action='store_true', default=False)
    args = parser.parse_args()

    try:
        data = json.load(open(args.file, 'rb'))
    except:
        data = json.load(open(args.file, 'r'))

    if args.all:
        pprint.pprint(data)
    else:
        pprint.pprint(data['json_args'])
