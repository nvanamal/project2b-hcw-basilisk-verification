"""Research-only HCW rendezvous-to-hold-point baseline."""
import argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
MU=3.986004418e14
R_E=6378.1363e3
ALT=700e3
N=np.sqrt(MU/(R_E+ALT)**3)
def A_hcw(n=N):
    A=np.zeros((6,6)); A[:3,3:]=np.eye(3)
    A[3,0]=3*n*n; A[3,4]=2*n
    A[4,3]=-2*n
    A[5,2]=-n*n
    return A
def B_hcw():
    B=np.zeros((6,3)); B[3:,:]=np.eye(3); return B
def lqr_gain():
    A=A_hcw(); B=B_hcw()
    Q=np.diag([1/100**2]*3+[1/0.2**2]*3)
    R=np.diag([1/2e-4**2]*3)
    P=solve_continuous_are(A,B,Q,R)
    return np.linalg.solve(R,B.T@P)
def rk4(x,u,dt,A,B):
    f=lambda q:A@q+B@u
    k1=f(x); k2=f(x+dt*k1/2); k3=f(x+dt*k2/2); k4=f(x+dt*k3)
    return x+dt*(k1+2*k2+2*k3+k4)/6
def run(seed=42, dt=2.0, duration=7200.0, noise_pos=0.2, noise_vel=0.0002,
        thrust_error=0.01, direction_error=0.005):
    rng=np.random.default_rng(seed); A=A_hcw(); B=B_hcw()
    goal=np.array([100.,0.,0.,0.,0.,0.])
    x=np.array([1000.,250.,-100.,-0.03,0.0,0.0])
    rows=[]; dv=0.; aborted=False; dwell=0.
    for k,t in enumerate(np.arange(0,duration+dt,dt)):
        xhat=x.copy(); xhat[:3]+=rng.normal(0,noise_pos,3); xhat[3:]+=rng.normal(0,noise_vel,3)
        natural=(A@xhat)[3:]
        kp=2.0e-5; kd=1.2e-2
        u_cmd=-natural-kp*(xhat[:3]-goal[:3])-kd*(xhat[3:]-goal[3:])
        umax=2e-2; norm=np.linalg.norm(u_cmd)
        if norm>umax: u_cmd*=umax/norm
        u_real=u_cmd*(1+rng.normal(0,thrust_error,3))+rng.normal(0,direction_error*umax,3)
        sep=np.linalg.norm(x[:3])
        if sep<50.0:
            aborted=True; u_real=np.zeros(3)
        rows.append([t,*x,*xhat,*u_cmd,*u_real,sep,dv,aborted])
        if aborted: break
        x=rk4(x,u_real,dt,A,B); dv+=np.linalg.norm(u_real)*dt
        pos_err=np.linalg.norm(x[:3]-goal[:3]); speed=np.linalg.norm(x[3:])
        dwell = dwell+dt if (pos_err<=5 and speed<=0.05 and np.linalg.norm(x[:3])>=50) else 0
        if dwell>=60: break
    cols=['t','x','y','z','vx','vy','vz','xhat','yhat','zhat','vxhat','vyhat','vzhat',
          'ux_cmd','uy_cmd','uz_cmd','ux_real','uy_real','uz_real','separation','delta_v','aborted']
    df=pd.DataFrame(rows,columns=cols); df.to_csv('trajectory.csv',index=False)
    final=df.iloc[-1]
    summary={'seed':seed,'success':bool(not aborted and dwell>=60),'aborted':bool(aborted),
             'final_time_s':float(final.t),'final_position_error_m':float(np.linalg.norm(final[['x','y','z']].to_numpy(float)-goal[:3])),
             'final_speed_mps':float(np.linalg.norm(final[['vx','vy','vz']].to_numpy(float))),
             'minimum_separation_m':float(df.separation.min()),'realized_delta_v_mps':float(final.delta_v)}
    with open('run_summary.json','w') as f: json.dump(summary,f,indent=2)
    fig,ax=plt.subplots(figsize=(7,6)); ax.plot(df.x,df.y,label='Chaser'); ax.scatter([0],[0],c='red',label='Debris')
    ax.scatter([100],[0],c='green',label='100 m hold point'); c=plt.Circle((0,0),50,color='red',alpha=.15); ax.add_patch(c)
    ax.set_aspect('equal','box'); ax.set_xlabel('Hill x (m)'); ax.set_ylabel('Hill y (m)'); ax.set_title('HCW rendezvous baseline'); ax.grid(); ax.legend(); fig.tight_layout(); fig.savefig('trajectory.png',dpi=180)
    print(json.dumps(summary,indent=2))
    return summary
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--controller',default='lqr',help='Legacy unused option; actual controller is HCW-compensated PD'); ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args(); run(seed=args.seed)
