import math
import random


class Joint:

    def scale_to_joint(self, x: float):
        return x * (self.upper - self.lower) + self.lower

    def scale_to_inputs(self, x: float):
        return (x - self.lower) / (self.upper - self.lower)

    def __init__(self, data, model, limits, name: str, tolerance: float):
        self.actuator = data.actuator(name)
        self.joint = data.joint(f"{name}_joint")
        self.link = model.body(f"{name}_link")
        self.index = self.link.id
        self.mass = self.link.mass[0]
        self.lower = limits[0]
        self.upper = limits[1]
        self.tolerance = tolerance

    def move(self, target: float):
        self.actuator.ctrl = self.scale_to_joint(target)
        self.joint.qvel = 0
        self.joint.qacc = 0

    def snap(self, target: float):
        target = self.scale_to_joint(target)
        self.actuator.ctrl = target
        self.joint.qpos = target
        self.joint.qvel = 0
        self.joint.qacc = 0

    def snap_random(self):
        target = self.scale_to_joint(random.uniform(0, 1))
        self.actuator.ctrl = target
        self.joint.qpos = target
        self.joint.qvel = 0
        self.joint.qacc = 0

    def target(self):
        return self.actuator.ctrl[0]

    def value(self):
        return self.joint.qpos[0]

    def velocity(self):
        return self.joint.qvel[0]

    def acceleration(self):
        return self.joint.qacc[0]

    def reached_target(self):
        return math.isclose(0, self.joint.qvel, abs_tol=self.tolerance)
