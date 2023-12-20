#!/usr/bin/env python3

import re
import json
import sys

def parse_testcase(input_data):
    power = re.match("total_package_power_watts: (?P<power>(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)", input_data)
    corner = re.search("corner: \"(?P<corner>\S+)\"", input_data)
    value = [["Name", "Value"], ["Total Package Power [W]", power["power"]], ["Corner", corner["corner"]]]
    return [{
        "name": "power measurement",
        "group": "performance",
        "type": "table",
        "value": value,
    }]

if __name__ == "__main__":
    input_data = sys.stdin.read()
    data = parse_testcase(input_data)

    print(json.dumps(data))
