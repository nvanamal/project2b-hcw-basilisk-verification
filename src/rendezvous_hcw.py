"""New HCW reconstruction, not original research source."""
import numpy as np

def rhs(s, u, n):
    x,y,z,vx,vy,vz=s
    return np.array([vx,vy,vz,3*n*n*x+2*n*vy+u[0],-2*n*vx+u[1],-n*n*z+u[2]])

def command(s,n,kp=2e-5,kd=.012,limit=.02):
    x,y,z,vx,vy,vz=s
    u=np.array([-kp*(x-100)-kd*vx-2*n*vy-3*n*n*x,-kp*y-kd*vy+2*n*vx,-kp*z-kd*vz+n*n*z])
    norm=np.linalg.norm(u)
    return u*min(1,limit/norm) if norm else u

def step(s,u,n,dt=2):
    a=rhs(s,u,n); b=rhs(s+dt*a/2,u,n)
    c=rhs(s+dt*b/2,u,n); d=rhs(s+dt*c,u,n)
    return s+dt*(a+2*b+2*c+d)/6

def run(seed=0,noisy=False):
    rng=np.random.default_rng(seed)
    n=np.sqrt(3.986004418e14/(6378136.3+700000)**3)
    s=np.array([1000.,250.,-100.,-.03,0.,0.])
    dwell=dv=0.; minimum=np.linalg.norm(s[:3])
    for t in range(0,7201,2):
        err=np.linalg.norm(s[:3]-[100,0,0]); speed=np.linalg.norm(s[3:])
        minimum=min(minimum,np.linalg.norm(s[:3]))
        success=dwell>=60 and minimum>=50
        if success or minimum<50 or t==7200:
            return dict(seed=seed,success=success,completion_s=t,position_error_m=err,speed_m_s=speed,minimum_separation_m=minimum,delta_v_m_s=dv)
        measured=s+np.r_[rng.normal(0,.2,3),rng.normal(0,.0002,3)] if noisy else s
        u=command(measured,n)
        if noisy: u=u*(1+rng.normal(0,.01,3))+rng.normal(0,.0001,3)
        s=step(s,u,n); dv+=2*np.linalg.norm(u)
        dwell=dwell+2 if err<=5 and speed<=.05 else 0
if __name__=="__main__":
    print(run())
