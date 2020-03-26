import jsondiff
import json

file1 = 'expectedData.jsonl'
file2 = 'MultiRC_val.jsonl'

file1json = {}
file1json["output"] = []
with open(file1) as file:
    file1data = file.readlines()
    for line in file1data:
        file1json["output"].append(line.strip())

file2json = {}
file2json["output"] = []
with open(file2) as file:
    file2data = file.readlines()
    for line in file2data:
        file2json["output"].append(line.strip())

output_file1 = 'new1.json'
output_file2 = 'new2.json'

with open(output_file1, 'w') as output_file:
    output_file.write(str(file1json).replace("'","").replace("output",'"output"'))

with open(output_file2, 'w') as output_file:
    output_file.write(str(file2json).replace("'","").replace("output",'"output"'))
