"""Generate NEW reconstruction results, not archived observations."""
import pandas as pd
from rendezvous_hcw import run
from pathlib import Path
if __name__ == '__main__':
    Path('outputs').mkdir(exist_ok=True)
    pd.DataFrame([run(i,True) for i in range(100)]).to_csv('outputs/hcw_reconstruction.csv',index=False)
