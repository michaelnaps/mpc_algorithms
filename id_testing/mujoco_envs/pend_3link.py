import numpy as np
from gym import utils
# from gym.envs.mujoco import mujoco_env
from mujoco_envs import mujoco_env


class Pend3LinkEnv(mujoco_env.MujocoEnv, utils.EzPickle):
	def __init__(self, dynamics_randomization=False):
		self.freeze = 0
		self.desired_vel = np.random.uniform(0.5, 1.5)
		mujoco_env.MujocoEnv.__init__(self, 'pend_3link.xml', 1)
		utils.EzPickle.__init__(self)
		self.count = 0

	def step(self, a):
		self.do_simulation(a, self.frame_skip)
		reward = 0
		ob = self._get_obs()
		done = False
		return ob, reward, done, {}



	def _get_obs(self):
		return np.concatenate([
			self.sim.data.qpos.flat[0:],
			np.clip(self.sim.data.qvel.flat, -100, 100)
		])

	def reset(self):
		init_pos = np.array([0.7076, 1.7264, -0.8632])
		init_vel = np.array([0.0, 0.0, 0.0])
    
		# init_pos = np.array([-0.1428,  0.7615, 0,  2.7045,  0.5034,  3.2426,  0.3292])
		# init_vel = np.array([0.7498, -0.0050, 0,  0.2315,  1.4370, -2.0440,  4.2029])

		  #qpos = init_pos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
		#qvel = init_vel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
		qpos = init_pos
		qvel = init_vel
		self.set_state(qpos, qvel)
		self.desired_vel = np.random.uniform(0.5, 1.5)

		# for i in range(1000):
		# 	self.step(np.zeros(4))
		# 	self.render()


		# next_state = np.concatenate((self._get_obs(), [self.desired_vel]),axis=0)

		# return next_state
		return self._get_obs()

	def stable_reward(self, desired_vel, velocity, angle):
		if abs(velocity - desired_vel) <= 0.1: #and data.desired_vel_x_fil > 0.01 and data.vel_x_b_fil > 0.01:
			reward = np.clip(0.01/abs(velocity - desired_vel + 0.0000001),0,1)
		else:
			reward = - 1*abs(velocity - desired_vel)
		if abs(angle - 0.18) < 0.2:
			reward += 0.5

		return reward

	def viewer_setup(self):
		self.viewer.cam.trackbodyid = 2
		self.viewer.cam.distance = self.model.stat.extent * 1.25
		self.viewer.cam.lookat[2] += .6
		self.viewer.cam.elevation = -10

	#============= This are new added methods. For using them you have to declare them in "mujoco_env.py" and "core.py"
	#============= Otherwise you have to call them from the main .py file by using "env.unwrapped.new_method_name()"
	def get_state(self):
		return self.sim.data.qpos, self.sim.data.qvel

	def get_sensor_data(self,sensor_name):
		return self.sim.data.get_sensor(sensor_name)    

	def assign_desired_vel(self,desired_vel):
		self.desired_vel = desired_vel
	#=====================================================================================================================
