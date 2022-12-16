import math
import random

import mujoco
import matplotlib.pyplot as plt

from mujoco import viewer


def clamp(target: float, lower: float, upper: float):
    return max(min(target, upper), lower)


class Robot:
    def __init__(self, path: str, tolerance: float):
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model)
        self.model.opt.gravity = 0
        self.tolerance = tolerance
        self.joints = []
        for i in range(self.model.nu):
            self.joints.append(Joint(self.data, self.model, self.model.actuator_ctrlrange[i], self.model.actuator(i).name, self.tolerance))

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

    def move(self, targets, duration: float = 100):
        index = 0
        for joint in self.joints:
            joint.move(targets[index])
            index += 1
        max_steps = int(duration / self.model.opt.timestep)
        step = 0
        while step < max_steps:
            step += 1
            mujoco.mj_step(self.model, self.data)
            if self.reached_targets():
                return True, step * self.model.opt.timestep
        return False, duration

    def snap(self, targets):
        index = 0
        for joint in self.joints:
            joint.snap(targets[index])
            index += 1
        mujoco.mj_forward(self.model, self.data)

    def random_move(self, duration: float = 100):
        targets = []
        for joint in self.joints:
            targets.append(joint.random_value())
        return self.move(targets, duration)

    def random_snap(self):
        targets = []
        for joint in self.joints:
            targets.append(joint.random_value())
        return self.snap(targets)

    def reached_targets(self):
        for joint in self.joints:
            if not joint.reached_target():
                return False
        return True

    def end_position(self):
        return self.data.xpos[self.joints[-1].index] - self.data.xpos[self.joints[0].index], self.data.xquat[self.joints[-1].index]

    def get_values(self):
        values = []
        for joint in self.joints:
            values.append(joint.value())
        return values

    def generate_pose(self):
        while True:
            self.random_snap()
            if self.reached_targets():
                return self.get_values()

    def generate_pair(self):
        inputs = self.generate_pose()
        pos, rot = self.end_position()
        inputs.append(pos[0])
        inputs.append(pos[1])
        inputs.append(pos[2])
        inputs.append(rot[0])
        inputs.append(rot[1])
        inputs.append(rot[2])
        inputs.append(rot[3])
        outputs = "TODO"
        return {"inputs": inputs, "outputs": outputs}

    def generate_dataset(self, size: int):
        data = []
        for i in range(size):
            data.append(self.generate_pair())

    def view(self):
        viewer.launch(self.model, self.data)

    def image(self):
        self.renderer.update_scene(self.data)
        plt.imshow(self.renderer.render())
        plt.show()

    def __str__(self):
        index = 1
        s = ""
        for joint in self.joints:
            s += f"Joint {index}\n" \
                 f"Target:\t\t{joint.target()}\n" \
                 f"Value:\t\t{joint.value()}\n" \
                 f"Reached:\t{joint.reached_target()}\n" \
                 f"Limits:\t\t[{joint.lower},{joint.upper}]\n" \
                 f"Velocity:\t{joint.velocity()}\n" \
                 f"Acc:\t\t{joint.acceleration()}\n" \
                 f"Mass:\t\t{joint.mass}\n" \
                 f"Position:\t{self.data.xpos[joint.index] - self.data.xpos[self.joints[0].index]}\n" \
                 f"Rotation:\t{self.data.xquat[joint.index]}\n"
            index += 1
        return s

    def move_debug(self, targets, duration: float = 100):
        reached, time = self.move(targets, duration)
        if reached:
            print(f"Reached targets in {time} seconds.")
        else:
            print(f"Could not reach targets in {time} seconds.")
        return reached, time

    def random_move_debug(self, duration: float = 100):
        reached, time = self.random_move(duration)
        if reached:
            print(f"Reached random targets in {time} seconds.")
        else:
            print(f"Could not reach targets in {time} seconds.")
        return reached, time


class Joint:
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
        self.actuator.ctrl = clamp(target, self.lower, self.upper)
        self.joint.qvel = 0
        self.joint.qacc = 0

    def snap(self, target: float):
        target = clamp(target, self.lower, self.upper)
        self.actuator.ctrl = target
        self.joint.qpos = target
        self.joint.qvel = 0
        self.joint.qacc = 0

    def snap_current(self):
        self.snap(self.actuator.ctrl)

    def random_value(self):
        return random.uniform(self.lower, self.upper)

    def target(self):
        return self.actuator.ctrl[0]

    def value(self):
        return self.joint.qpos[0]

    def velocity(self):
        return self.joint.qvel[0]

    def acceleration(self):
        return self.joint.qacc[0]

    def reached_target(self):
        return math.isclose(self.actuator.ctrl, self.joint.qpos, abs_tol=self.tolerance)
