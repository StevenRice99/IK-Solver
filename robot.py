import math

import mujoco
import numpy as np

from mujoco import viewer

from joint import Joint


def point_distance(a, b):
    return math.sqrt(pow(b[0] - a[0], 2) + pow(b[1] - a[1], 2) + pow(b[2] - a[2], 2))


def quaternion_inverse(q):
    q[1] = -q[1]
    q[2] = -q[2]
    q[3] = -q[3]
    return q

class Robot:
    def __init__(self, path: str, tolerance: float = 0.001):
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model)
        self.model.opt.gravity = 0
        self.tolerance = tolerance
        self.joints = []
        self.target = self.model.body("target")
        for i in range(self.model.nu):
            self.joints.append(Joint(self.data, self.model, self.model.actuator_ctrlrange[i], self.model.actuator(i).name, self.tolerance))
        self.chain_length = 0
        for i in range(1, len(self.joints)):
            self.chain_length += point_distance(self.data.xpos[self.joints[i - 1].index], self.data.xpos[self.joints[i].index])
        self.set_end_target()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

    def perform(self, duration: float = 100):
        max_steps = int(duration / self.model.opt.timestep)
        step = 0
        while step < max_steps:
            step += 1
            mujoco.mj_step(self.model, self.data)
            if self.reached_targets():
                return True, step * self.model.opt.timestep
        return False, duration

    def move(self, targets, duration: float = 100):
        index = 0
        for joint in self.joints:
            joint.move(targets[index])
            index += 1
        return self.perform(duration)

    def snap(self, targets):
        index = 0
        for joint in self.joints:
            joint.snap(targets[index])
            index += 1
        return self.perform()

    def snap_random(self):
        index = 0
        for joint in self.joints:
            joint.snap_random()
            index += 1
        return self.perform()

    def reached_targets(self):
        for joint in self.joints:
            if not joint.reached_target():
                return False
        return True

    def joint_position(self, joint):
        return self.data.xpos[joint.index] - self.data.xpos[self.joints[0].index]

    def joint_rotation(self, joint):
        return quaternion_inverse(self.data.xquat[self.joints[0].index]) * self.data.xquat[joint.index]

    def local_end_position(self):
        return self.joint_position(self.joints[-1]), self.joint_rotation(self.joints[-1])

    def global_end_position(self):
        return self.data.xpos[self.joints[-1].index], self.data.xquat[self.joints[-1].index]

    def accuracy(self):
        p, q = self.global_end_position()
        d = point_distance(self.target.pos, p)
        b = math.sqrt((self.chain_length + d) * (point_distance(self.data.xpos[self.joints[0].index], p) + d))
        return (0 if b == 0 else math.pi * d / b) + (2 * math.acos(max(0, min(1, np.dot(q, self.target.quat)))))

    def set_end_target(self):
        self.target.pos, self.target.quat = self.global_end_position()

    def get_values(self):
        values = []
        for joint in self.joints:
            values.append(joint.value())
        return values

    def generate_pair(self):
        self.snap_random()
        pos, rot = self.local_end_position()
        relative_pos = pos / self.chain_length
        inputs = [relative_pos[0], relative_pos[1], relative_pos[2], rot[0], rot[1], rot[2], rot[3]]
        self.snap_random()
        starting = self.get_values()
        inputs.extend(starting)
        outputs = "TODO"
        return {"INPUTS": inputs, "OUTPUTS": outputs}

    def generate_dataset(self, size: int):
        data = []
        for i in range(size):
            data.append(self.generate_pair())

    def view(self, targets=None):
        if targets is not None:
            index = 0
            for joint in self.joints:
                joint.move(targets[index])
                index += 1
        viewer.launch(self.model, self.data)

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
                 f"Position:\t{self.joint_position(joint)}\n" \
                 f"Rotation:\t{self.joint_rotation(joint)}\n"
            index += 1
        return s

    def move_debug(self, targets, duration: float = 100):
        reached, time = self.move(targets, duration)
        if reached:
            print(f"Reached targets in {time} seconds.")
        else:
            print(f"Could not reach targets in {time} seconds.")
        return reached, time
