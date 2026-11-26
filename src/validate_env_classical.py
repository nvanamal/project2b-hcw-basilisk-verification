"""Classical environment parity test; no learned controller."""
import numpy as np
from safe_rendezvous_env import (SafeRendezvousEnv, HOLD_POINT, MEAN_MOTION_RAD_S, POSITION_GAIN, VELOCITY_GAIN, MAXIMUM_CONTROL_ACCELERATION_MPS2)

def classical_acceleration(state):
    position=state[:3]
    velocity=state[3:]
    position_error=position-HOLD_POINT[:3]
    x,z=position[0],position[2]
    vx,vy,vz=velocity
    n=MEAN_MOTION_RAD_S
    acceleration=np.zeros(3,dtype=np.float64)
    acceleration[0]=-POSITION_GAIN*position_error[0]-VELOCITY_GAIN*vx-2.0*n*vy-3.0*n*n*x
    acceleration[1]=-POSITION_GAIN*position_error[1]-VELOCITY_GAIN*vy+2.0*n*vx
    acceleration[2]=-POSITION_GAIN*position_error[2]-VELOCITY_GAIN*vz+n*n*z
    magnitude=np.linalg.norm(acceleration)
    if magnitude>MAXIMUM_CONTROL_ACCELERATION_MPS2:
        acceleration*=MAXIMUM_CONTROL_ACCELERATION_MPS2/magnitude
    return acceleration

def main():
    environment=SafeRendezvousEnv(use_uncertainty=False)
    observation,info=environment.reset(seed=42)
    total_reward=0.0
    while True:
        acceleration=classical_acceleration(environment.state)
        action=acceleration/MAXIMUM_CONTROL_ACCELERATION_MPS2
        observation,reward,terminated,truncated,info=environment.step(action)
        total_reward+=reward
        if terminated or truncated:
            break
    print(info)
    print("Accumulated reward:",total_reward)

if __name__=="__main__":
    main()
