"""Original execution filename: monte_carlo.py."""
import argparse
import pandas as pd
from rendezvous_hcw import run
ap=argparse.ArgumentParser(); ap.add_argument('--runs',type=int,default=100); args=ap.parse_args()
rows=[]
for seed in range(args.runs):
    s=run(seed=seed); rows.append(s)
pd.DataFrame(rows).to_csv('monte_carlo_results.csv',index=False)
print(pd.DataFrame(rows).describe(include='all'))
