from abc import ABC
import numpy as np

from envs.panda_env import PandaEnv


class PushEnv(PandaEnv, ABC):
    """
    Dynamic pushing environment in Drake
    """
    def __init__(self, 
                dt=0.002,
                render=False,
                visualize_contact=False,
                camera_param=None,
                hand_type='plate',
                diff_ik_filter_hz=-1,
                contact_solver='sap',
                panda_joint_damping=200,
                ):
        super(PushEnv, self).__init__(
            dt=dt,
            render=render,
            visualize_contact=visualize_contact,
            camera_param=camera_param,
            hand_type=hand_type,
            diff_ik_filter_hz=diff_ik_filter_hz,
            contact_solver=contact_solver,
            panda_joint_damping=panda_joint_damping,
        )
        self.finger_init_pos = 0.055
        self.bottle_initial_pos = np.array([0.45, 0.0, 0.05 + self.table_offset])


    def reset_task(self, task):
        return NotImplementedError


    @property
    def parameter(self):
        return [self.task['obj_mu'], 
                self.task['obj_modulus'], 
                self.task['obj_com_x'],
                self.task['obj_com_y']]


    def load_objects(self, task=None):
        # Load veggies - more like templates - save body and frame ids
        bottle_model_index, bottle_body_indice = \
            self.station.AddModelFromFile(
                '/examples/panda/data/bottle.sdf',
                name='bottle',
            )
        self.bottle_body = self.plant.get_body(bottle_body_indice[0])
        self.bottle_default_inertia = self.bottle_body.default_spatial_inertia() 


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

        # Set table properties
        self.set_obj_dynamics(context_inspector, 
                              sg_context, 
                              self.table_body,
                              hc_dissipation=1.0,
                              sap_dissipation=0.1,
                              mu=0.1,
                              hydro_modulus=7,
                              hydro_resolution=0.1, # does not matter
                              compliance_type='rigid')

        # Set hand properties
        self.set_obj_dynamics(context_inspector, 
                              sg_context, 
                              self.hand_body,
                              hc_dissipation=1.0,
                              sap_dissipation=0.1,
                              mu=0.3,
                              hydro_modulus=6,
                              hydro_resolution=0.1, # does not matter
                              compliance_type='rigid')

        # Set bottle properties
        self.set_obj_dynamics(context_inspector, 
                              sg_context, 
                              self.bottle_body,
                              hc_dissipation=1.0,
                              sap_dissipation=0.1,
                              mu=task['obj_mu'],
                              hydro_modulus=task['obj_modulus'],
                              hydro_resolution=0.002,  # matters
                              compliance_type='compliant')

        # First, revert inertia back to origin
        self.bottle_body.SetSpatialInertiaInBodyFrame(plant_context, self.bottle_default_inertia)

        # Next, shift the current inertia to new COM - need to shift in the opposite direction.
        inertia = self.bottle_body.CalcSpatialInertiaInBodyFrame(plant_context)
        inertia = inertia.Shift([-task['obj_com_x'], -task['obj_com_y'], 0])
        self.bottle_body.SetSpatialInertiaInBodyFrame(plant_context, inertia)

        # Reset bottle
        self.set_body_pose(self.bottle_body, plant_context, 
                            p=self.bottle_initial_pos,
                            rpy=[0, 0, 0])
        self.set_body_vel(self.bottle_body, plant_context)
        station_context = self.station.GetMyContextFromRoot(context)
        return self._get_obs(station_context)


    def _get_obs(self, station_context):
        if not self.flag_use_camera:
            return None
        color_image = self.color_image_port.Eval(station_context).data[:,:,:3] # HxWx4
        color_image = np.transpose(color_image, [2,0,1])
        depth_image = np.squeeze(self.depth_image_port.Eval(station_context).data)
        
        # Normalize
        depth_image = ((self.camera_params['max_depth']-depth_image)/(self.camera_params['max_depth']-self.camera_params['min_depth']))*255
        depth_image = np.uint8(depth_image)[np.newaxis]
        image = np.vstack((color_image, depth_image))
        return image


    def _get_bottle_vel(self, plant_context):
        return self.plant.EvalBodySpatialVelocityInWorld(
            plant_context, 
            self.bottle_body,
            ).get_coeffs().reshape(6,1)


    def _get_bottle_pose(self, plant_context):
        return self.plant.EvalBodyPoseInWorld(plant_context, self.bottle_body)
