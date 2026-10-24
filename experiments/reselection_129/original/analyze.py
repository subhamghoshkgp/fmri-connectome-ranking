import argparse
from szv3.report import analyze
p=argparse.ArgumentParser()
p.add_argument('results')
a=p.parse_args()
r=analyze(a.results)
print('Complete:',r['complete'],'Data:',r['kind'])
