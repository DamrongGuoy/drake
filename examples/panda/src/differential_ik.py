import numpy as np

from pydrake.all import LeafSystem, BasicVector, JacobianWrtVariable


class PseudoInverseController(LeafSystem):


    def __init__(self, plant, dt):
        LeafSystem.__init__(self)
        self._dt = dt
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._panda = plant.GetModelInstanceByName("panda")
        self._G = plant.GetBodyByName("panda_link8").body_frame()
        self._W = plant.world_frame()

        self.V_G_port = self.DeclareVectorInputPort("V_WG", BasicVector(6))
        self.q_port = self.DeclareVectorInputPort("q", BasicVector(7))
        self.qdot_port = self.DeclareVectorInputPort("qdot", BasicVector(7))
        self.DeclareVectorOutputPort("panda_velocity", BasicVector(7), 
                                     self.CalcOutput, 
                                     {self.V_G_port.ticket()})
        self.panda_start = plant.GetJointByName("panda_joint1").velocity_start()
        self.panda_end = plant.GetJointByName("panda_joint7").velocity_start()

        # Prev solution for initial guess
        self.prev_sol = np.zeros((7))
        self.prev_v_g = np.zeros((6))
        self.prev_qdot = np.zeros((6))


    def CalcOutput(self, context, output):
        # print('{:.17f}'.format(context.get_time()), context.num_total_states(), hex(id(context)))
        # self._plant_context = self._cache.plant_context

        V_G = self.V_G_port.Eval(context)
        q = self.q_port.Eval(context)
        qdot = self.qdot_port.Eval(context)
        # prev_V_G = self.V_port.Eval(context)[9].get_coeffs()
        # print('Input: ', V_G)
        # print('Cur qdot: ', qdot)
        # print('Cur vel: ', prev_V_G)

        # Return zero if V_G all zero - do not calculate
        if np.all(np.abs(V_G) < 1e-5):
            output.SetFromVector(np.zeros((7)))
            return

        # Solve if first time in a time step - otherwise just return the last result
        # if flag_first:
        self.q = q
        self.qdot = qdot
    
        # Set states for controller plant?
        self._plant.SetPositions(self._plant_context, self._panda, self.q)
        self._plant.SetVelocities(self._plant_context, self._panda, self.qdot)

        # Update jacobian and current spatial velocities
        self.J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context, JacobianWrtVariable.kV, 
            self._G, [0,0,0], self._W, self._W) # ang, then trans
        self.J_G = self.J_G[:,self.panda_start:self.panda_end+1]
        self.cur_v = self.J_G.dot(self.qdot)

        # Option 1: psuedo-inverse
        # v = np.linalg.pinv(self.J_G).dot(V_G)
        # output.SetFromVector(v)

        # Option 2: damped psuedo-inverse
        damping = np.eye((6))*0.002
        pinv = self.J_G.T.dot(np.linalg.inv(self.J_G.dot(self.J_G.T) + damping))
        v = pinv.dot(V_G)

        # Update
        output.SetFromVector(v)
        self.prev_sol = v
        self.prev_v_g = V_G
        self.prev_qdot = qdot
