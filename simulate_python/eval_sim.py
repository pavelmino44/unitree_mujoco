"""MuJoCo simulator for the scenario evaluation (driven by ros2_ws/src/go2_eval).

It is the simulator of unitree_mujoco.py (same robot model, same time step SIMULATE_DT, same DDS bridge
UnitreeSdk2Bridge: lowstate / lowcmd / sportmodestate, so go2_policy_runner sees exactly the robot it sees in
unitree_mujoco.py), plus what an evaluation needs and the interactive simulator does not have:

  * the scene of one evaluation scenario (eval_scene.py: the exact IsaacLab terrain tile + border);
  * a control connection (TCP, JSON lines) to the evaluation: reset the robot to a pose, hold the base in
    place while the controller stands up, release it, hold it again;
  * the ground truth of every control step (0.02 s by default): base pose and velocity, joint state and
    torques, and the contact forces of the robot's body parts (base, head, legs) with the terrain.

The simulation runs in real time, like unitree_mujoco.py, because go2_policy_runner runs on the wall clock.

Normally started by go2_eval. By hand (it waits for the evaluation to connect on --port):

    python eval_sim.py --catalog ../eval_scenarios/catalog.json --scenario flat --seed 42 --port 47011 --viewer

Protocol (one JSON object per line).
  sim -> eval:  {"type": "hello", ...model info...}
                {"type": "sample", "t": ..., ...}        every --sample_dt of simulated time while released
                {"type": "ack", "cmd": ..., "t_sim": ...}
  eval -> sim:  {"cmd": "reset", "pos": [x, y, z], "quat": [w, x, y, z], "joints": {"FL_hip_joint": q, ...}}
                    robot to this pose, zero velocity, base held in place
                {"cmd": "release"}   free the base; the samples start (t = 0)
                {"cmd": "hold"}      hold the base where it is; the samples stop
                {"cmd": "quit"}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import socket
import sys
import threading
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config  # noqa: E402  (unitree_mujoco simulator configuration)
import eval_scene  # noqa: E402

# Parts of the Go2 body (MuJoCo geoms) whose contacts are recorded. In MuJoCo the head is part of the
# base_link body (in IsaacLab it is the separate Head_upper / Head_lower bodies): the base_link collision
# geoms in front of HEAD_X are the head. The feet (spheres at the end of the calf) are not recorded.
TRUNK_BODY = "base_link"
HEAD_X = 0.25  # m, base_link frame
LEG_BODY_SUFFIXES = ("_hip", "_thigh", "_calf")
FOOT_GEOMS = ("FL", "FR", "RL", "RR")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="MuJoCo simulator of the go2_eval scenario evaluation.")
    ap.add_argument("--catalog", required=True, help="catalog.json written by export_mujoco_scenarios.py")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1", help="address of the evaluation (control connection)")
    ap.add_argument("--port", type=int, required=True, help="port of the evaluation (control connection)")
    ap.add_argument("--robot", default=config.ROBOT)
    ap.add_argument("--domain", type=int, default=config.DOMAIN_ID, help="DDS domain (default: config.py)")
    ap.add_argument("--interface", default=config.INTERFACE, help="DDS network interface (default: config.py)")
    ap.add_argument("--dt", type=float, default=config.SIMULATE_DT, help="physics time step (default: config.py)")
    ap.add_argument("--sample_dt", type=float, default=0.02, help="time between two samples (the control step)")
    ap.add_argument("--markers", default="[]", help="JSON list of visual ground markers (see eval_scene.scene_xml)")
    ap.add_argument("--hfield_resolution", type=float, default=eval_scene.HFIELD_RESOLUTION)
    ap.add_argument("--viewer", action="store_true", help="open the MuJoCo viewer")
    ap.add_argument("--save_scene", default=None, help="also write the scene XML here")
    return ap.parse_args(argv)


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world vector v into the body frame of the unit quaternion q = (w, x, y, z)."""
    w, x, y, z = q
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return r.T @ v


class EvalSim:
    def __init__(self, args):
        self.args = args
        catalog = eval_scene.load_catalog(args.catalog)
        terrain = eval_scene.load_terrain(eval_scene.terrain_path(catalog, args.scenario, args.seed))
        xml, self.terrain_info = eval_scene.scene_xml(
            terrain, args.robot, markers=json.loads(args.markers), hfield_resolution=args.hfield_resolution
        )
        self.model = eval_scene.load_model(xml, args.robot, save_as=args.save_scene)
        self.model.opt.timestep = args.dt
        self.data = mujoco.MjData(self.model)
        self.lock = threading.Lock()
        self.running = True

        m = self.model
        self.base_body = m.body(TRUNK_BODY).id
        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free) != 1:
            raise RuntimeError("the robot must have exactly one free joint")
        self.free_qpos = int(m.jnt_qposadr[free[0]])
        self.free_dof = int(m.jnt_dofadr[free[0]])
        self.robot_bodies = {b for b in range(m.nbody) if b != 0 and self._root_of(b) == self.base_body}
        self.mass = float(sum(m.body_mass[b] for b in self.robot_bodies))
        # actuators -> the dof they drive (for the power)
        self.act_dof = np.array([m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
        self.joint_dofs = np.array(sorted({int(m.jnt_dofadr[j]) for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE}))
        self.parts = self._classify_geoms()
        self.n_sub = max(1, int(round(args.sample_dt / args.dt)))

        # episode state
        self.hold_pose = self.data.qpos[self.free_qpos:self.free_qpos + 7].copy()
        self.holding = True
        self.released_at = None
        self.steps = 0
        self.part_force = {p: 0.0 for p in ("base", "head", "legs")}
        self.prev_dq = None
        self.joint_acc = 0.0
        self.lag = {"overruns": 0, "worst_s": 0.0}

        self.out = queue.Queue()
        self.sock = None

    def _root_of(self, b: int) -> int:
        while self.model.body_parentid[b] != 0:
            b = self.model.body_parentid[b]
        return b

    def _classify_geoms(self) -> dict:
        """geom id -> part ("base", "head", "legs", "foot") for the collision geoms of the robot."""
        m = self.model
        parts = {}
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if b not in self.robot_bodies or (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0):
                continue
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if name in FOOT_GEOMS or name.endswith("_foot") or body.endswith("_foot"):
                parts[g] = "foot"
            elif b == self.base_body:
                parts[g] = "head" if m.geom_pos[g][0] > HEAD_X else "base"
            elif body.endswith(LEG_BODY_SUFFIXES):
                parts[g] = "legs"
        return parts

    # ---- control connection --------------------------------------------------------------------------

    def connect(self):
        self.sock = socket.create_connection((self.args.host, self.args.port), timeout=10.0)
        self.sock.settimeout(None)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        threading.Thread(target=self._sender, daemon=True).start()
        threading.Thread(target=self._receiver, daemon=True).start()
        m = self.model
        def names(ids):  # unnamed geoms: <body>/<type>#<index in the body>
            out = []
            for g in ids:
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                if not name:
                    b = int(m.geom_bodyid[g])
                    k = g - int(m.body_geomadr[b])
                    name = f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)}/{mujoco.mjtGeom(m.geom_type[g]).name[7:].lower()}#{k}"
                out.append(name)
            return out

        self.send(
            {
                "type": "hello",
                "scenario": self.args.scenario,
                "seed": self.args.seed,
                "dt": self.args.dt,
                "sample_dt": self.n_sub * self.args.dt,
                "mass": self.mass,
                "joint_names": [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.dof_jntid[d]) for d in self.joint_dofs],
                "parts": {p: names([g for g, q in self.parts.items() if q == p]) for p in ("base", "head", "legs", "foot")},
                "terrain": self.terrain_info,
                "mujoco": mujoco.__version__,
                "viewer": bool(self.args.viewer),
            }
        )

    def send(self, obj: dict):
        self.out.put(obj)

    def _sender(self):
        while True:
            obj = self.out.get()
            try:
                self.sock.sendall((json.dumps(obj) + "\n").encode())
            except OSError:
                self.running = False
                return

    def _receiver(self):
        buf = b""
        while self.running:
            try:
                chunk = self.sock.recv(65536)
            except OSError:
                chunk = b""
            if not chunk:
                self.running = False
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    self._handle(json.loads(line))

    def _handle(self, msg: dict):
        cmd = msg.get("cmd")
        with self.lock:
            d, m = self.data, self.model
            if cmd == "reset":
                mujoco.mj_resetData(m, d)
                q = self.free_qpos
                d.qpos[q:q + 3] = msg["pos"]
                d.qpos[q + 3:q + 7] = msg["quat"]
                for name, value in msg.get("joints", {}).items():
                    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
                    if j >= 0:
                        d.qpos[m.jnt_qposadr[j]] = value
                d.qvel[:] = 0.0
                mujoco.mj_forward(m, d)
                self.hold_pose = d.qpos[q:q + 7].copy()
                self.holding = True
                self.released_at = None
            elif cmd == "hold":
                self.hold_pose = d.qpos[self.free_qpos:self.free_qpos + 7].copy()
                self.holding = True
                self.released_at = None
            elif cmd == "release":
                self.holding = False
                self.released_at = d.time
                self.steps = 0
                self.part_force = {p: 0.0 for p in self.part_force}
                self.prev_dq = d.qvel[self.joint_dofs].copy()
                self.lag = {"overruns": 0, "worst_s": 0.0}
                self._wall0 = time.perf_counter()
            elif cmd == "quit":
                self.running = False
            self.send({"type": "ack", "cmd": cmd, "t_sim": d.time})

    # ---- physics -----------------------------------------------------------------------------------

    def _hold(self):
        q, v = self.free_qpos, self.free_dof
        self.data.qpos[q:q + 7] = self.hold_pose
        self.data.qvel[v:v + 6] = 0.0

    def _contacts(self):
        """Largest net contact force (N) on a body of each part during this physics step (robot vs. the world)."""
        m, d = self.model, self.data
        net: dict[tuple, np.ndarray] = {}
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            p1, p2 = self.parts.get(g1), self.parts.get(g2)
            r1 = int(m.geom_bodyid[g1]) in self.robot_bodies
            r2 = int(m.geom_bodyid[g2]) in self.robot_bodies
            if r1 == r2:  # self-collision or world-world
                continue
            part, g, sign = (p1, g1, -1.0) if r1 else (p2, g2, 1.0)  # the force acts on geom2 along +normal
            if part not in self.part_force:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            f = sign * (c.frame.reshape(3, 3).T @ f6[:3])
            key = (part, int(m.geom_bodyid[g]))
            net[key] = net.get(key, 0.0) + f
        for (part, _), f in net.items():
            self.part_force[part] = max(self.part_force[part], float(np.linalg.norm(f)))

    def _sample(self) -> dict:
        m, d = self.model, self.data
        q, v = self.free_qpos, self.free_dof
        quat = d.qpos[q + 3:q + 7].copy()
        lin_w = d.qvel[v:v + 3].copy()
        ang_b = d.qvel[v + 3:v + 6].copy()  # free joint: angular velocity in the body frame
        dq = d.qvel[self.joint_dofs]
        tau = d.actuator_force
        power = float(np.sum(np.abs(tau * d.qvel[self.act_dof])))
        sample = {
            "type": "sample",
            "k": self.steps // self.n_sub,
            "t": d.time - self.released_at,
            "pos": d.qpos[q:q + 3].tolist(),
            "quat": quat.tolist(),
            "lin_vel_w": lin_w.tolist(),
            "lin_vel_b": quat_rotate_inverse(quat, lin_w).tolist(),
            "ang_vel_b": ang_b.tolist(),
            "power": power,
            "joint_acc": self.joint_acc,
            "joint_vel_max": float(np.max(np.abs(dq))) if len(dq) else 0.0,
            "forces": dict(self.part_force),
            "lag": dict(self.lag),
            "wall": time.perf_counter() - self._wall0,
        }
        self.part_force = {p: 0.0 for p in self.part_force}
        return sample

    def step(self):
        with self.lock:
            if self.holding:
                self._hold()
            mujoco.mj_step(self.model, self.data)
            if self.holding:
                self._hold()
                mujoco.mj_forward(self.model, self.data)
                return
            dq = self.data.qvel[self.joint_dofs]
            self.joint_acc = float(np.linalg.norm((dq - self.prev_dq) / self.model.opt.timestep))
            self.prev_dq = dq.copy()
            self._contacts()
            self.steps += 1
            if self.steps % self.n_sub == 0:
                self.send(self._sample())

    def physics_loop(self, viewer=None):
        dt = self.model.opt.timestep
        next_t = time.perf_counter()
        while self.running and (viewer is None or viewer.is_running()):
            self.step()
            next_t += dt
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.1:  # far behind real time: count it and restart the schedule
                with self.lock:
                    self.lag["overruns"] += 1
                    self.lag["worst_s"] = max(self.lag["worst_s"], -delay)
                next_t = time.perf_counter()
        self.running = False


def main(argv=None):
    args = parse_args(argv)
    sim = EvalSim(args)
    print(
        f"[eval_sim] {args.scenario} seed {args.seed}: {sim.terrain_info['kind']} terrain, dt={args.dt}, "
        f"robot mass {sim.mass:.2f} kg, DDS domain {args.domain} on '{args.interface}'",
        flush=True,
    )

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py_bridge import UnitreeSdk2Bridge

    ChannelFactoryInitialize(args.domain, args.interface)
    bridge = UnitreeSdk2Bridge(sim.model, sim.data)  # noqa: F841  (publishes lowstate, applies lowcmd)
    sim.connect()

    if args.viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = sim.base_body
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = -130
        physics = threading.Thread(target=sim.physics_loop, args=(viewer,), daemon=True)
        physics.start()
        while viewer.is_running() and sim.running:
            with sim.lock:
                viewer.sync()
            time.sleep(config.VIEWER_DT)
        sim.running = False
        physics.join(timeout=1.0)
        viewer.close()
    else:
        sim.physics_loop()
    # the DDS threads of the bridge are not daemons: leave without waiting for them
    time.sleep(0.05)
    os._exit(0)


if __name__ == "__main__":
    main()
