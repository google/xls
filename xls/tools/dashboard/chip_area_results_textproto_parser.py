#!/usr/bin/env python3

import re
import json
import sys

def parse_testcase(input_data):
    area = re.match("area_micro_meters_squared: (?P<area>\d+)", input_data)
    value = [["Name", "Value"], ["Chip area [um^2]", area["area"]]]
    return [{
        "name": "area measurement",
        "group": "performance",
        "type": "table",
        "value": value,
    }]

if __name__ == "__main__":
    data = sys.stdin.read()
    json_data = parse_testcase(data)
    print(json.dumps(json_data))
