from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc import autoargs
import os
import numpy as np

class MultiLinkReacherEnv(MujocoEnv, Serializable):
    """
    Adapted from Gym's reacher environment, but has multiple links and lies in
        3D.
    Observation: joint angle and goal location
    Action: joint angle actuations
    Reward: negative distance to the goal location
    Termination: after reaching close enough to the goal
    """

    def __init__(
            self,
            deterministic=True,
            dist_threshold=0.01,
            n_links=4,
            link_length=0.1,
            *args,
            **kwargs
        ):
        """
        :param dist_threshold: if the dist to goal is under the threshold,
            terminate
        :param deterministic: fix the initial reacher position and velocity and
            the goal position
        """
        self.frame_skip = 2 # default in OpenAI
        self.deterministic = deterministic
        self.dist_threshold = dist_threshold
        self.goal = np.array((0.,0.,0.2))
        self.file_path = os.path.join(
            os.path.dirname(__file__),
            "models",
            "multilink_reacher_temp.xml",
        )
        self.n_links = n_links
        self.link_length = link_length
        if not hasattr(self, "xml_generated"):
            self.xml_generated = False
        self.generate_xml_file()
        kwargs["file_path"] = self.file_path
        super().__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())
        self.window_config = dict(
            title="billiards",
            xpos=0,
            ypos=0,
            width=500,
            height=500,
        )

    @overrides
    def render(self,close=False, config=None):
        super().get_viewer(config=self.window_config)
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 1
        self.viewer.loop_once()
        if close:
            self.stop_viewer()

    def step(self, action):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        dist = np.linalg.norm(vec)
        reward_dist = - dist
        reward_ctrl = - 1e-5 * np.square(action).sum() # some coefficient?
        reward = reward_dist + reward_ctrl
        self.forward_dynamics(action)
        ob = self.get_current_obs()
        done = dist < self.dist_threshold
        return ob, reward, done, dict(
                reward_dist=reward_dist,
                reward_ctrl=reward_ctrl,
                fingertip=self.get_body_com("fingertip"),
            )

    def log_stats(self, algo, epoch, paths, ax):
        # plot trajectories
        for path in paths:
            fingertips = path["env_infos"]["fingertip"]
            xs = fingertips[:,0]
            ys = fingertips[:,1]
            zs = fingertips[:,2]
            ax.plot(xs=xs,ys=ys,zs=zs)
            ax.plot(xs=[xs[0]], ys=[ys[0]], zs=[zs[0]],
                marker='o', color='k')

        # plot and label the goal
        gx = self.goal[0]
        gy = self.goal[1]
        gz = self.goal[2]
        ax.plot(xs=[gx], ys=[gy], zs=[gz], marker='*', color='r')
        ax.text(gx, gy, gz, '(%.2f, %.2f, %.2f)'%(gx, gy, gz))

        # set axes properties
        lim = self.link_length * self.n_links
        ax.set_xlim3d(-lim, lim)
        ax.set_ylim3d(-lim, lim)
        ax.set_zlim3d(-lim, lim)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # log useful stats
        dist_cost = np.concatenate([
            -path["env_infos"]["reward_dist"]
            for path in paths
        ])
        ctrl_cost = np.concatenate([
            -path["env_infos"]["reward_ctrl"]
            for path in paths
        ])
        stats = {
            'env: DistCostAverage': np.mean(dist_cost),
            'env: DistCostMax': np.max(dist_cost),
            'env: DistCostMin': np.min(dist_cost),
            'env: CtrlCostAverage': np.mean(ctrl_cost),
            'env: CtrlCostMax': np.max(ctrl_cost),
            'env: CtrlCostMin': np.min(ctrl_cost),
        }
        return stats


    @overrides
    def reset_mujoco(self, init_state=None):
        if self.deterministic:
            qpos = np.copy(self.init_qpos)
        else:
            qpos = self.init_qpos + np.random.uniform(
                low=-0.1, high=0.1, size=(self.model.nq, 1))
        qpos[-3:,0] = self.goal

        if self.deterministic:
            qvel = np.copy(self.init_qvel)
        else:
            qvel = self.init_qvel + \
                np.random.uniform(low=-.005, high=.005, size=(self.model.nv, 1))
        qvel[-3:] = 0

        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
        self.model.data.qacc = self.init_qacc
        self.model.data.ctrl = self.init_ctrl
        return self.get_current_obs()

    def get_current_obs(self):
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

    def generate_xml_file(self):
        if self.xml_generated:
            return
        from lxml.etree import Element as ET
        from lxml import etree

        n_links = self.n_links

        # create XML
        root = ET('mujoco', model="multilink_reacher")
        compiler = ET('compiler', angle="radian", inertiafromgeom="true")
        root.append(compiler)

        default = ET('default')
        default.append(ET('joint', armature="1", damping="10", limited="true"))
        default.append(ET('geom', contype="0", friction="1 0.1 0.1", rgba="0.7 0.7 0 1"))
        root.append(default)

        option = ET('option', gravity="0 0 0", integrator="RK4", timestep="0.01")
        root.append(option)

        worldbody = ET('worldbody')
        cur_link = worldbody
        for i in range(n_links):
            body = ET(
                'body',
                name="link_%d"%(i),
                pos="%.2f 0 0"%(self.link_length) if i > 0 else "0 0 0",
            )
            body.append(ET("joint",
                limited="false", name="link_%d_ball"%(i), pos="0 0 0", type="ball"))
            body.append(ET("geom",
                fromto="0 0 0 0.1 0 0", name="link_%d_geom"%(i),
                rgba="0.0 0.4 0.6 1", size=".01", type="capsule"
            ))
            if i == n_links - 1:
                fingertip = ET(
                    "body",
                    name="fingertip",
                    pos="0.11 0 0"
                )
                fingertip.append(ET(
                    "geom",
                    contype="0", name="fingertip", pos="0 0 0",
                    rgba="0.0 0.8 0.6 1", size=".01", type="sphere"
                ))
                body.append(fingertip)
            cur_link.append(body)
            cur_link = body
        target = ET("body", name="target", pos="0.1 -0.1 0.01")
        target.append(ET(
            "joint",
            armature="0", axis="1 0 0", damping="0",
            limited="true", name="target_x", pos="0 0 0",
            range="-.27 .27", ref="-.1", stiffness="0", type="slide"
        ))
        target.append(ET(
            "joint",
            armature="0", axis="0 1 0", damping="0",
            limited="true", name="target_y", pos="0 0 0",
            range="-.27 .27", ref="-.1", stiffness="0", type="slide"
        ))
        target.append(ET(
            "joint",
            armature="0", axis="0 0 1", damping="0",
            limited="true", name="target_z", pos="0 0 0",
            range="-.27 .27", ref="-.1", stiffness="0", type="slide"
        ))
        target.append(ET(
            "geom",
            conaffinity="0", contype="0", name="target",
            pos="0 0 0", rgba="0.9 0.2 0.2 1", size=".009", type="sphere"
        ))
        worldbody.append(target)

        root.append(worldbody)

        actuator = ET('actuator')
        for i in range(n_links):
            joint_name = "link_%d_ball"%(i)
            for gear in ["200 0 0", "0 200 0", "0 0 200"]: # Euler angles?
                motor = ET(
                    'motor',
                    joint=joint_name,
                    ctrllimited="true",
                    ctrlrange="-1.0 1.0",
                    gear=gear,
                )
                actuator.append(motor)
        root.append(actuator)

        # pretty string
        s = etree.tostring(root, pretty_print=True).decode()
        print(s)
        with open(self.file_path, "w") as f:
            f.write(s)
        self.xml_generated = True

    # Disallow parallel workers to regenerated the xml file,
    # because if one is writing while the other is building the model,
    # errors will occur.
    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d.update({
            "xml_generated": self.xml_generated,
        })
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self.xml_generated = d["xml_generated"]
