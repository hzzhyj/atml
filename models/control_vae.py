from math import exp
from models.beta_vae import *

class PIDControl():
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Ik1 = 0.0
        self.Wk1 = 1.0
        self.ek1 = 0.0

    def compute_Kp_term(self, err, scale = 1):
        return self.Kp * 1.0 / (1.0 + float(scale) * exp(err))

    def pid_update(self, expected_KL, KL):
        err_k = expected_KL - KL
        Pk = self.compute_Kp_term(err_k) + 1
        Ik = self.Ik1 + self.Ki * err_k

        if(self.Wk1 < 1):
            Ik = self.Ik1

        Wk = Pk + Ik
        self.Wk1 = Wk
        self.Ik1 = Ik

        if(Wk < 1):
            Wk = 1

        return Wk, err_k

class ControlVAEDSprites(BetaVAEDSprites):
    def __init__(self, n_latents=10, controller_args):
        super(BetaVAEDSprites, self).__init__(n_latents)

        self.init_controller(controller_args)

    def init_controller(self, controller_args):
        self.C = controller_args['C']
        self.C_max = controller_args['C_max']
        self.C_step_val = controller_args['C_step_val']
        self.C_step_period = controller_args['C_step_period']
        self.beta_controller = PIDControl(controller_args['Kp'], controller_args['Ki'], controller_args['Kd'])

    def update_beta(self, iter_idx, KL):
        # Update C
        if((iter_idx % self.C_step_period) == 0):
            self.C += self.C_step_val

        # Threshold C
        if self.C > self.C_max:
            self.C = self.C_max

        # Get new beta
        beta, _ self.beta_controller.pid_update(self.C, KL)
        return beta
