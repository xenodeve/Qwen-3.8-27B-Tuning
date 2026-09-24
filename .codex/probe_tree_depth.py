"""Supplementary integer-tree depth probe, separate from frozen acceptance tests."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,'C:/AI/qwen38-tuning/bench')
from run_bench import extract_code

ap=argparse.ArgumentParser()
ap.add_argument('responses',type=Path)
a=ap.parse_args()
row=next(json.loads(line) for line in a.responses.read_text(encoding='utf-8').splitlines()
         if json.loads(line)['case']=='session-tree_codec')
code=extract_code(row['response']['choices'][0]['message'].get('content') or '')
probe='''
import json
root = Node(0)
node = root
for i in range(1, 1500):
    node.left = Node(-i)
    node = node.left
try:
    restored = deserialize(serialize(root))
    node = restored
    count = 0
    while node is not None:
        assert node.val == -count and node.right is None
        count += 1
        node = node.left
    assert count == 1500
    print(json.dumps({'depth':1500,'round_trip':True}))
except Exception as error:
    print(json.dumps({'depth':1500,'round_trip':False,'error':type(error).__name__}))
'''
r=subprocess.run([sys.executable,'-I','-c',code+'\n'+probe],capture_output=True,text=True,timeout=20)
out={'source':str(a.responses),'kind':'supplementary depth robustness; original functional score unchanged',
     'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
(a.responses.parent/'tree-depth-probe.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
