import pickle
from pydrake.all import FindResourceOrThrow
import numpy as np

from envs.push_traj_env import PushTrajEnv


if __name__ == '__main__':
    # Load tasks
    dataset = FindResourceOrThrow("drake/examples/panda/data/bottle_task_v2_demo.pkl")
    print("= Loading tasks from", dataset)
    with open(dataset, 'rb') as f:
        task_all = pickle.load(f)
    
    # Load trajectories
    traj_path = FindResourceOrThrow("drake/examples/panda/data/10ms_push_traj.pkl")
    with open(traj_path, 'rb') as f:
        data = pickle.load(f)
    traj_all = data['traj_all']
    x_endpoint_all = data['x_endpoint_all']
    q_endpoint_all = data['q_endpoint_all']
    bin_index_all = data['bin_index_all']

    # Run
    env = PushTrajEnv(
                    dt=0.01,
                    render=True,
                    visualize_contact=True,
                    # diff_ik_filter_hz=-1,
                    contact_solver='sap',
                    panda_joint_damping=1.0,
                    )
    for ind in range(1, 100):
        task = task_all[ind]
        print('')
        print('Resetting...')
        print(f"Task - modulus: {10**task['obj_modulus']} - friction coefficient: {task['obj_mu']}")
        print('')
        obs = env.reset(task=task)
            
        # Randomly sample a trajectory
        traj_index = np.random.randint(0, 2401)
        dq_d = traj_all[traj_index]
        qstar = q_endpoint_all[traj_index][0]
        action = (dq_d, qstar)
        
        # Execute
        _, _, _, info = env.step(action)
