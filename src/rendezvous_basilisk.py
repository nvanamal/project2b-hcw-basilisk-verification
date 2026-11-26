"""Compact transcription of author-supplied rendezvous_basilisk_validated_backup.py.
Not byte-identical; diagnostic plotting omitted. Execution verification pending.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from Basilisk.simulation import spacecraft, extForceTorque
from Basilisk.utilities import SimulationBaseClass, macros, orbitalMotion, simIncludeGravBody

OUTPUT_DIRECTORY=Path(__file__).resolve().parent
def create_spacecraft(tag,mass_kg):
    obj=spacecraft.Spacecraft()
    obj.ModelTag=tag
    obj.hub.mHub=mass_kg
    obj.hub.r_BcB_B=[[0.],[0.],[0.]]
    obj.hub.IHubPntBc_B=[[500.,0.,0.],[0.,500.,0.],[0.,0.,500.]]
    obj.hub.sigma_BNInit=[[0.],[0.],[0.]]
    obj.hub.omega_BN_BInit=[[0.],[0.],[0.]]
    return obj

def frame(r,v):
    radial=r/np.linalg.norm(r)
    h=np.cross(r,v)
    normal=h/np.linalg.norm(h)
    along=np.cross(normal,radial)
    return np.vstack((radial,along/np.linalg.norm(along),normal)),h/np.dot(r,r)

def relative(r,v,rd,vd):
    C,w=frame(r,v)
    dr=rd-r
    return C@dr,C@(vd-v-np.cross(w,dr)),C

def run_basilisk_case(seed=42,noise_pos=0.,noise_vel=0.,thrust_error=0.,direction_error=0.,save_outputs=True,make_figure=False,quiet=False,worker_output=None):
    rng=np.random.default_rng(seed)
    sim=SimulationBaseClass.SimBaseClass()
    proc=sim.CreateNewProcess("dynamicsProcess")
    task="dynamicsTask"
    proc.addTask(sim.CreateNewTask(task,macros.sec2nano(2.)))
    chief=create_spacecraft("chiefSpacecraft",500.)
    deputy=create_spacecraft("deputySpacecraft",250.)
    force=extForceTorque.ExtForceTorque()
    force.ModelTag="deputyControlForce"
    force.extForce_N=[[0.],[0.],[0.]]
    deputy.addDynamicEffector(force)
    gravity=simIncludeGravBody.gravBodyFactory()
    earth=gravity.createEarth()
    earth.isCentralBody=True
    gravity.addBodiesTo(chief); gravity.addBodiesTo(deputy)
    orbit=orbitalMotion.ClassicElements()
    orbit.a=6378.1363e3+700.e3
    orbit.e=orbit.i=orbit.Omega=orbit.omega=orbit.f=0.
    r,v=map(lambda a:np.asarray(a,dtype=float),orbitalMotion.elem2rv(earth.mu,orbit))
    C,w=frame(r,v)
    dr=C.T@np.array([1000.,250.,-100.])
    rd=r+dr
    vd=v+C.T@np.array([-.03,0.,0.])+np.cross(w,dr)
    for obj,p,q in [(chief,r,v),(deputy,rd,vd)]:
        obj.hub.r_CN_NInit=p.reshape(3,1).tolist()
        obj.hub.v_CN_NInit=q.reshape(3,1).tolist()
        sim.AddModelToTask(task,obj)
    sim.AddModelToTask(task,force)
    cr=chief.scStateOutMsg.recorder(); drc=deputy.scStateOutMsg.recorder()
    sim.AddModelToTask(task,cr); sim.AddModelToTask(task,drc)
    sim.InitializeSimulation()
    n=np.sqrt(earth.mu/orbit.a**3)
    goal=np.array([100.,0.,0.])
    t=dv=dwell=0.
    success=aborted=False
    reason="maximum simulation time reached"
    commands=[]
    while t<7200.:
        p,q,C=relative(r,v,rd,vd)
        sep=np.linalg.norm(p)
        if sep<50. or sep>50000.:
            aborted=True
            reason="keep-out-zone violation" if sep<50. else "maximum separation exceeded"
            break
        dwell=dwell+2. if np.linalg.norm(p-goal)<=5. and np.linalg.norm(q)<=.05 else 0.
        if dwell>=60.:
            success=True; reason="hold-point requirements satisfied"; break
        pm=p+rng.normal(0.,noise_pos,3)
        qm=q+rng.normal(0.,noise_vel,3)
        x,y,z=pm; vx,vy,vz=qm
        u=np.array([-2e-5*(x-100.)-.012*vx-2*n*vy-3*n**2*x,
                    -2e-5*y-.012*vy+2*n*vx,
                    -2e-5*z-.012*vz+n**2*z])
        mag=np.linalg.norm(u)
        if mag>.02: u=u*(.02/mag)
        real=u*(1.+rng.normal(0.,thrust_error,3))+rng.normal(0.,direction_error*.02,3)
        F=250.*(C.T@real)
        force.extForce_N=F.reshape(3,1).tolist()
        commands.append([t,*u,*F])
        dv+=np.linalg.norm(real)*2.
        t=min(t+2.,7200.)
        sim.ConfigureStopTime(macros.sec2nano(t)); sim.ExecuteSimulation()
        r=np.asarray(cr.r_BN_N[-1],dtype=float); v=np.asarray(cr.v_BN_N[-1],dtype=float)
        rd=np.asarray(drc.r_BN_N[-1],dtype=float); vd=np.asarray(drc.v_BN_N[-1],dtype=float)
    force.extForce_N=[[0.],[0.],[0.]]
    times=np.asarray(cr.times(),dtype=float)*macros.NANO2SEC
    rows=[]
    for ti,r,v,rd,vd in zip(times,cr.r_BN_N,cr.v_BN_N,drc.r_BN_N,drc.v_BN_N):
        r,v,rd,vd=map(lambda a:np.asarray(a,dtype=float),(r,v,rd,vd))
        p,q,_=relative(r,v,rd,vd)
        rows.append([ti,*r,*rd,*p,*q,np.linalg.norm(p),np.linalg.norm(p-goal),np.linalg.norm(q)])
    cols=["time_s"]+[f"chief_{a}_inertial_m" for a in "xyz"]+[f"deputy_{a}_inertial_m" for a in "xyz"]+[f"relative_{a}_hill_m" for a in "xyz"]+[f"relative_v{a}_hill_mps" for a in "xyz"]+["relative_distance_m","hold_point_error_m","relative_speed_mps"]
    df=pd.DataFrame(rows,columns=cols)
    last=df.iloc[-1]
    summary=dict(model="controlled Basilisk chief-deputy rendezvous",seed=int(seed),position_noise_sigma_m=float(noise_pos),velocity_noise_sigma_mps=float(noise_vel),thrust_error_sigma=float(thrust_error),additive_actuation_scale=float(direction_error),additive_actuation_sigma_mps2=float(direction_error*.02),success=success,aborted=aborted,termination_reason=reason,final_time_s=float(last.time_s),initial_relative_distance_m=float(df.relative_distance_m.iloc[0]),final_relative_distance_m=float(last.relative_distance_m),final_position_error_m=float(last.hold_point_error_m),final_relative_speed_mps=float(last.relative_speed_mps),minimum_separation_m=float(df.relative_distance_m.min()),minimum_hold_point_error_m=float(df.hold_point_error_m.min()),realized_delta_v_mps=float(dv),hold_point_hill_m=goal.tolist(),position_tolerance_m=5.,speed_tolerance_mps=.05,required_hold_time_s=60.,keep_out_radius_m=50.,maximum_control_acceleration_mps2=.02)
    if save_outputs:
        df.to_csv(OUTPUT_DIRECTORY/"controlled_basilisk_trajectory.csv",index=False)
        pd.DataFrame(commands,columns=["time_s"]+[f"commanded_a{a}_hill_mps2" for a in "xyz"]+[f"commanded_f{a}_inertial_n" for a in "xyz"]).to_csv(OUTPUT_DIRECTORY/"controlled_basilisk_commands.csv",index=False)
        (OUTPUT_DIRECTORY/"controlled_basilisk_summary.json").write_text(json.dumps(summary,indent=4),encoding="utf-8")
    if worker_output: Path(worker_output).write_text(json.dumps(summary,indent=4),encoding="utf-8")
    if not quiet: print(json.dumps(summary,indent=4))
    return summary

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=42)
    for name in ["noise-pos","noise-vel","thrust-error","direction-error"]: p.add_argument("--"+name,type=float,default=0.)
    p.add_argument("--worker-output")
    a=p.parse_args()
    run_basilisk_case(**vars(a))
