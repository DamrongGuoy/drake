import pickle
from pydrake.all import FindResourceOrThrow
import numpy as np

from envs.scoop_env import ScoopEnv


if __name__ == '__main__':
    # Load tasks - 100 of different ones
    dataset = FindResourceOrThrow("drake/examples/panda/data/cylinder_ellipsoid_tasks.pkl")
    print("= Loading tasks from", dataset)
    with open(dataset, 'rb') as f:
        task_all = pickle.load(f)

    # Run
    env = ScoopEnv(dt=0.005,
                   render=True,
                   visualize_contact=True,  # conflict with swapping geometry
                   hand_type='panda',
                   camera_param=None,
                   diff_ik_filter_hz=200,
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

        action = np.array([5, 0.6, -0.2, -0.6])
        action[0] *= np.pi/180
        obs, reward, _, info = env.step(action)
