"""Analyze supplied matched CSVs; no synthetic fallback."""
import argparse
import numpy as np
import pandas as pd
from scipy import stats
def analyze(h,b):
    if len(h)!=100 or len(b)!=100 or not h.seed.is_unique or not b.seed.is_unique or set(h.seed)!=set(b.seed):
        raise ValueError("Expected 100 unique matched seeds per branch")
    p=h.merge(b,on="seed",suffixes=("_h","_b"),validate="one_to_one")
    rows=[]
    for m in ["completion_s","position_error_m","speed_m_s","minimum_separation_m","delta_v_m_s"]:
        d=p[m+"_b"]-p[m+"_h"]
        if not np.isfinite(d).all(): raise ValueError("Missing/nonfinite metric")
        sd=d.std(ddof=1); mean=d.mean(); half=stats.t.ppf(.975,99)*sd/10
        rows.append(dict(metric=m,mean_difference=mean,ci_low=mean-half,ci_high=mean+half,paired_t_p=stats.ttest_1samp(d,0).pvalue if sd else np.nan,wilcoxon_p=stats.wilcoxon(d).pvalue if (d!=0).any() else 1.,cohen_dz=mean/sd if sd else np.nan))
    return pd.DataFrame(rows)
if __name__=="__main__":
    a=argparse.ArgumentParser(); a.add_argument("hcw"); a.add_argument("basilisk"); a.add_argument("output"); x=a.parse_args()
    analyze(pd.read_csv(x.hcw),pd.read_csv(x.basilisk)).to_csv(x.output,index=False)
