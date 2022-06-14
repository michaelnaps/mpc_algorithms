import sys

sys.path.insert(0, 'test/models/.');

from statespace_alip import *
from statespace_3link import *
from modeuler import *

class InputsALIP:
    def __init__(self, prev_input):
        self.gravity_acc          = -9.81;
        self.joint_masses         = [40];
        self.link_lengths         = [0.95];
        self.CP_maxdistance       = 0.1;
        self.input_bounds         = [100, 500];
        self.prev_input           = prev_input;

class Inputs3link:
    def __init__(self):
        self.num_inputs           = 3;
        self.gravity_acc          = -9.81;
        self.damping_coefficients = [0, 0, 0];
        self.joint_masses         = [5, 5, 30];
        self.link_lengths         = [0.5, 0.5, 0.6];

if __name__ == "__main__":
    inputs_alip  = InputsALIP([0,0]);
    inputs_3link = Inputs3link();
    (T_alip, q_alip) = modeuler(2, 1, 100, 0.025, statespace_alip, [0, 0], [1, 0], inputs_alip);
    (T_3link, q_3link) = modeuler(6, 1, 100, 0.025, statespace_3link, [1.57, 0, 0, 0, 0, 0], [-10, 0, 0], inputs_3link);

    animation_alip(T_alip, q_alip, inputs_alip);
    animation_3link(T_3link, q_3link, inputs_3link);
