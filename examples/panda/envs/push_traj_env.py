from abc import ABC
import numpy as np

from pydrake.all import PiecewisePolynomial
from envs.push_env import PushEnv


class PushTrajEnv(PushEnv, ABC):
    """
    Dynamic pushing environment in Drake
    """
    def __init__(self, 
                 dt=0.005,
                 render=False,
                 visualize_contact=False,
                 camera_param=None,
                 hand_type='plate',
                 diff_ik_filter_hz=-1,
                 contact_solver='sap',
                 panda_joint_damping=1.0,
                 ):
        super(PushTrajEnv, self).__init__(
            dt=dt,
            render=render,
            visualize_contact=visualize_contact,
            camera_param=camera_param,
            hand_type=hand_type,
            diff_ik_filter_hz=diff_ik_filter_hz,
            contact_solver=contact_solver,
            panda_joint_damping=panda_joint_damping,
        )
        # Default goal
        self._goal = np.array([0.80, 0.0])
        self._goal_base = np.array([0.45, 0])    # bottle init

        # Scaling - max distance to goal
        self.max_dist_bottle_goal = 0.5


    @property
    def goal(self,) :
        return self._goal


    @goal.setter
    def goal(self, value):
        self._goal = value + self._goal_base


    def _get_obs(self, station_context=None):
        return np.array([], dtype=np.single)


    # only function overriding
    def step(self, action):
        """
        Initialize simulator and then execute open-loop.
        """
        # Get new context
        context = self.simulator.get_mutable_context()
        station_context = self.station.GetMyContextFromRoot(context)
        plant_context = self.plant.GetMyContextFromRoot(context)
        controller_plant_context = self.controller_plant.GetMyContextFromRoot(context)
        sg_context = self.sg.GetMyMutableContextFromRoot(context)
        query_object = self.sg.get_query_output_port().Eval(sg_context)

        # Extract action
        dq_d, qstar = action

        # Reset EE
        self.set_arm(plant_context, qstar)

        # Initialize state interpolator/integrator
        self.reset_state(plant_context, context)

        # Reset simulation
        sim_context = self.simulator.get_mutable_context()
        sim_context.SetTime(0.)
        self.simulator.Initialize()

        ######################## Run #######################

        # Push forward - cartesian velocity and diff IK
        end_step = dq_d.shape[0]
        self.plant.SetVelocities(plant_context, self.panda, np.zeros((7)))
        self.reset_state(plant_context, context)

        for step in range(1, end_step):
            context = self.simulator.get_mutable_context()
            plant_context = self.plant.GetMyContextFromRoot(context)
            self.ik_result_port.FixValue(station_context, np.zeros((7)))
            self.V_WG_command_port.FixValue(station_context, np.zeros((6)))
            self.V_J_command_port.FixValue(station_context, dq_d[step])

            # Simulate forward
            t = step*self.dt
            try:
                status = self.simulator.AdvanceTo(t)
            except RuntimeError as e:
                print('Sim error with task: ', self.task)
                return self._get_obs(), 0, True, {'error': True, 'bottle_p_final': [0.5, 0, 0], 'reward': 0}    #? ideally there should be no error so we don't have to put an arbitrary bottle_p_final to avoid error

        # Rest - assume bottle stops by the end
        context = self.simulator.get_mutable_context()
        plant_context = self.plant.GetMyContextFromRoot(context)
        dura = 0.5
        init_step = end_step
        end_step = int(dura / self.dt) + init_step
        vel_init = self.get_ee_vel(plant_context)
        vel_1 = np.array([[0], [0], [0], 
                          [0], [0], [0]])
        traj_V_G = PiecewisePolynomial.FirstOrderHold([0, dura], 
                                                np.hstack((vel_init, vel_1)))
        t_init = t
        for step in range(init_step, end_step):
            context = self.simulator.get_mutable_context()
            plant_context = self.plant.GetMyContextFromRoot(context)
            self.ik_result_port.FixValue(station_context, np.zeros((7)))
            self.V_WG_command_port.FixValue(station_context, 
                                            traj_V_G.value(t - t_init))
            self.V_J_command_port.FixValue(station_context, np.zeros((7)))

            # Simulate forward
            t = step*self.dt
            try:
                status = self.simulator.AdvanceTo(t)
            except RuntimeError as e:
                print('Sim error with task: ', self.task)
                return self._get_obs(), 0, True, {'error': True, 'bottle_p_final': [0.5, 0, 0], 'reward': 0}

        # Get reward
        context = self.simulator.get_mutable_context()
        plant_context = self.plant.GetMyContextFromRoot(context)
        reward = 0

        # Always done: single step
        done = True
        return np.array([]), reward, done, {}
