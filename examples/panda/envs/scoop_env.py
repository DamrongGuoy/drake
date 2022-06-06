import numpy as np
import os

from pydrake.all import RotationMatrix, RollPitchYaw, PiecewisePolynomial, RigidTransform, FixedOffsetFrame, CollisionFilterDeclaration
import scipy.interpolate

from envs.panda_env import PandaEnv


class ScoopEnv(PandaEnv):
    def __init__(self, 
                dt=0.002,
                render=False,
                camera_param=None,
                visualize_contact=False,
                hand_type='panda',
                diff_ik_filter_hz=200,
                contact_solver='sap',
                panda_joint_damping=1.0,
                ):
        super(ScoopEnv, self).__init__(
            dt=dt,
            render=render,
            camera_param=camera_param,
            visualize_contact=visualize_contact,
            hand_type=hand_type,
            diff_ik_filter_hz=diff_ik_filter_hz,
            contact_solver=contact_solver,
            panda_joint_damping=panda_joint_damping,
        )
        self.veggie_x = 0.65    # for scooping direction - assume veggies around this position in x
        self.finger_init_pos = 0.04
        self.spatula_init_z = 0.002

        # Fixed dynamics parameter
        self.veggie_hc_dissipation = 1.0
        self.veggie_hydro_resolution = 0.005


    def reset_task(self, task):
        return NotImplementedError


    def load_objects(self, task=None):
        # Load spatula
        self.spatula_model_index, self.spatula_body_index = \
            self.station.AddModelFromFile(
                        '/examples/panda/data/models/spatula_long/spatula_oxo_nylon_square_issue7322_low.sdf',
                        name='spatula',
                        )
        self.spatula_base = self.plant.get_body(self.spatula_body_index[0])  # only base
        self.spatula_base_frame = self.plant.GetFrameByName("origin", self.spatula_model_index)
        self.spatula_blade_frame = self.plant.GetFrameByName("spatula_blade_origin_frame", self.spatula_model_index)
        self.spatula_grasp_frame = self.plant.GetFrameByName("spatula_grasp_frame", self.spatula_model_index)
        self.spatula_tip_frame = self.plant.GetFrameByName("spatula_tip_frame", self.spatula_model_index)

        # Locate veggie path
        if task is None:
            veggie_path = '/examples/panda/data/veggie_2link.sdf'
        else:
            sdf_split = task['sdf'].split('/')
            veggie_path = os.path.join('/examples/panda/data', os.path.join(sdf_split[-2], sdf_split[-1]))

        # Load veggie template with fixed number of links (bodies to be replaced later) - save body and frame ids
        self.veggie_body_all = {}   # list of lists, in the order of links
        self.veggie_frame_all = {}
        for ind in range(self.task['obj_num']):
            veggie_model_index, veggie_body_indice = \
                self.station.AddModelFromFile(
                    veggie_path,
                    name='veggie'+str(ind),
                )
            self.veggie_body_all[ind] = [self.plant.get_body(index) for index in veggie_body_indice]    # all links
        self.veggie_base_all = [b[0] for b in self.veggie_body_all.values()]

        # Add a generic frame for veggies - fixed to table
        self.T_veggie = np.array([self.veggie_x, 0, self.table_offset])
        self.veggie_fixed_frame = self.plant.AddFrame(
            FixedOffsetFrame("veggie_fixed_frame", 
                            self.plant.world_frame(), 
                            RigidTransform(self.T_veggie)
                            )
            )


    def reset(self, task=None):
        """
        Call parent to reset arm and gripper positions (build if first-time). Reset veggies and task. Do not initialize simulator.
        """
        task = super().reset(task)

        # Get new context
        context = self.simulator.get_mutable_context()
        plant_context = self.plant.GetMyContextFromRoot(context)
        sg_context = self.sg.GetMyMutableContextFromRoot(context)
        query_object = self.sg.get_query_output_port().Eval(sg_context)
        context_inspector = query_object.inspector()

        # Set veggie geometry - bodies is a list of bodies for one piece
        sdf_cfg = task['sdf_cfg']
        sg_geometry_ids = context_inspector.GetAllGeometryIds()
        for _, bodies in self.veggie_body_all.items():
            for body_ind, body in enumerate(bodies):
                body_str = str(body_ind)

                # Change body dynamics
                self.set_obj_dynamics(context_inspector, 
                        sg_context,
                        body,
                        hc_dissipation=self.veggie_hc_dissipation,
                        sap_dissipation=0.1,
                        mu=self.task['obj_mu'],
                        hydro_modulus=self.task['obj_modulus'],
                        hydro_resolution=self.veggie_hydro_resolution,
                        compliance_type='compliant',
                        )

                # Change mass - not using, specified in sdf
                body.SetMass(plant_context, sdf_cfg['m'+body_str])

            # Exclude collision within body - use contac
            body_geometry_set = self.plant.CollectRegisteredGeometries(bodies)
            self.sg.collision_filter_manager(sg_context).Apply(
                CollisionFilterDeclaration().ExcludeWithin(body_geometry_set)
            )

        # Set table properties
        self.set_obj_dynamics(context_inspector, 
                              sg_context, 
                              self.table_body,
                              hc_dissipation=1.0,
                              sap_dissipation=0.1,
                              mu=0.3,
                              hydro_modulus=5,
                              hydro_resolution=0.1, # does not matter
                              compliance_type='compliant')

        # Change global params - time step for both plant and controller_plant? seems impossible
        # point - plant.set_penetration_allowance (stiffness of normal penalty forces)
        # point/hydro - plant.set_stiction_tolerance (threshold for sliding)

        # Move spatula away from veggies
        self.set_body_pose(self.spatula_base, plant_context, 
                           p=[0.3, 0, 0.01],
                           rpy=[0, 0, 0])
        self.set_body_vel(self.spatula_base, plant_context)

        # Set veggie pose from task
        for ind, veggie_base in enumerate(self.veggie_base_all):
            self.set_body_pose(veggie_base, plant_context, 
                                p=[task['obj_x'][ind],
                                   task['obj_y'][ind],
                                   task['obj_z'][ind]+self.table_offset,
                                   ],
                                rpy=[0, 0, 0])
            self.set_body_vel(veggie_base, plant_context)

        # Get fixed transforms between frames
        self.p_spatula_tip_to_spatula_base = self.spatula_base_frame.CalcPose(plant_context, self.spatula_tip_frame)
        self.p_spatula_tip_to_spatula_grasp = self.spatula_grasp_frame.CalcPose(plant_context, self.spatula_tip_frame)

        ######################## Observation ########################

        station_context = self.station.GetMyContextFromRoot(context)
        return np.empty(())


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
        inspector = query_object.inspector()

        # Extract action
        s_yaw = 0
        # s_yaw = action[0]
        s_pitch_init = action[0]
        # s_x_veggie_tip = action[2]
        s_x_veggie_tip = -0.02
        xd_1, pd_1, pd_2 = action[1:]

        # Veggit to tip
        R_VT = RotationMatrix(RollPitchYaw(0, 0, s_yaw))
        p_VT = [s_x_veggie_tip, 0., 0.]
        p_T = self.T_veggie + p_VT

        # Reset spatula between gripper - in real, we can make a holder for spatula, thus ensures grasp always at the same pose on the spatula. Make sure the tip of spatula touches the table when the spatula is tilted
        R_T = R_VT.multiply(RotationMatrix(RollPitchYaw(0, s_pitch_init, 0)))
        R_S = R_T.multiply(self.p_spatula_tip_to_spatula_base.rotation())
        p_S = p_T + R_T.multiply(self.p_spatula_tip_to_spatula_base.translation())
        #  + np.array([0,0,self.table_offset]
        self.set_body_pose(self.spatula_base, plant_context, p=p_S, rm=R_S)
        self.set_body_vel(self.spatula_base, plant_context)

        # Finally, EE
        R_TE = RotationMatrix(RollPitchYaw(0, -np.pi, np.pi/2))
        R_E = R_VT.multiply(R_TE)
        grasp_pose = self.spatula_grasp_frame.CalcPoseInWorld(plant_context)
        if self.hand_type == 'panda_foam':
            p_E = grasp_pose.translation() + np.array([0, 0, 0.16])
        elif self.hand_type == 'panda':
            p_E = grasp_pose.translation() + np.array([0, 0, 0.10])
        else:
            raise "Unknown hand type!"
        qstar = self.ik(plant_context, controller_plant_context, T_e=p_E, R_e=R_E)
        self.set_arm(plant_context, qstar)
        self.set_gripper(plant_context, self.finger_init_pos)

        # Initialize state interpolator/integrator
        self.reset_state(plant_context, context)

        # Reset simulation
        sim_context = self.simulator.get_mutable_context()
        sim_context.SetTime(0.)
        self.simulator.Initialize()

        ######################## Trajectory ########################
        q_all = []
        dq_all = []
        v_all = []
        v_d_all = []
        ddq_all = []
        dddq_all = []
        dq_prev = np.zeros((7))
        ddq_prev = np.zeros((7))
        hand_width_command = -0.2

        # Close gripper
        num_t_init = int(0.2 / self.dt)
        for t_ind in range(1, num_t_init):
            context = self.simulator.get_mutable_context()
            plant_context = self.plant.GetMyContextFromRoot(context)
            self.ik_result_port.FixValue(station_context, np.zeros((7)))
            self.V_WG_command_port.FixValue(station_context, [0]*6)  # hold arm in place
            self.hand_position_command_port.FixValue(station_context, 
                                                     hand_width_command)

            # Keep spatula in place
            self.set_body_pose(self.spatula_base, plant_context, 
                                    p=p_S,
                                    rm=R_S)
            self.set_body_vel(self.spatula_base, plant_context)

            # Simulate forward
            t = t_ind*self.dt
            try:
                status = self.simulator.AdvanceTo(t)
            except RuntimeError as e:
                print(f'Sim error at time {t}!')
                return self._get_obs(station_context), 0, True, {}

        # Time for spline 
        tn = [0, 0.25, 0.1, 0.25, 0.1, 0.1]
        tn = np.cumsum(tn)
        ts = np.arange(0, tn[-1], self.dt)

        # Spline for x direction
        xd = np.zeros((6))
        # xd[1:4] = action[3:6]
        xd[1] = xd_1
        xd[2] = xd_1 + 0.1
        xd[3] = 0.2
        poly_xd = scipy.interpolate.CubicSpline(tn, xd, bc_type='clamped')
        xds = poly_xd(ts)

        # Spline for pitch direction
        pitchd = np.zeros((6))
        # pitchd[2:4] = action[6:8]
        pitchd[1] = pd_1
        pitchd[2:4] = pd_2
        poly_pitchd = scipy.interpolate.CubicSpline(tn, pitchd, bc_type='clamped')
        pitchds = poly_pitchd(ts)

        # Spline for z direction
        zd = [0, -0.01, 0.02, 0.02, 0, 0]
        poly_zd = scipy.interpolate.CubicSpline(tn, zd, bc_type='clamped')
        zds = poly_zd(ts)

        # Go through trajectory
        num_t = len(ts)
        num_obs = 20
        obs_freq = int(num_t / num_obs)
        for t_ind in range(0, num_t):
            V_G = np.array([0, pitchds[t_ind], 0, 
                            xds[t_ind], 0, zds[t_ind]]).reshape(6, 1)
        
            context = self.simulator.get_mutable_context()
            plant_context = self.plant.GetMyContextFromRoot(context)
            hand_force_measure = self.hand_force_measure_port.Eval(station_context)
            self.ik_result_port.FixValue(station_context, np.zeros((7)))
            self.V_WG_command_port.FixValue(station_context, V_G)
            self.hand_position_command_port.FixValue(station_context, 
                                                    hand_width_command)
            # Simulate forward
            t = (t_ind+num_t_init)*self.dt
            try:
                status = self.simulator.AdvanceTo(t)
            except RuntimeError as e:
                print(f'Sim error at time {t}!')
                return self._get_obs(station_context), 0, True, {}


        # Get reward - veggie piece on spatula
        info = {}
        context = self.simulator.get_mutable_context()
        plant_context = self.plant.GetMyContextFromRoot(context)
        station_context = self.station.GetMyContextFromRoot(context)
        pose_blade = self.spatula_blade_frame.CalcPoseInWorld(plant_context)
        reward = 0
        for veggie_body in self.veggie_base_all:
            veggie_pose = self.plant.EvalBodyPoseInWorld(plant_context, veggie_body)
            blade_veggie = pose_blade.InvertAndCompose(veggie_pose)
            # .translation() - self.table_offset
            # veggie_vel = self.plant.EvalBodySpatialVelocityInWorld(plant_context, veggie_body).translational()
            if blade_veggie.translation()[2] > 0.001:
                reward += 1/self.task['obj_num']
            # info['obj_loc'] += list(veggie_pos[:2])
        info['reward'] = reward

        # Always done: single step
        done = True
        return np.array([]), reward, done, info


    def _get_obs(self, station_context=None):
        if not self.flag_use_camera:
            return np.array([], dtype=np.single)
        color_image = self.color_image_port.Eval(station_context).data[:,:,:3] # HxWx4
        color_image = np.transpose(color_image, [2,0,1])
        depth_image = np.squeeze(self.depth_image_port.Eval(station_context).data)

        # Normalize
        depth_image = ((self.camera_params['max_depth']-depth_image)/(self.camera_params['max_depth']-self.camera_params['min_depth']))*255
        depth_image = np.uint8(depth_image)[np.newaxis]
        image = np.vstack((color_image, depth_image))
        return image
