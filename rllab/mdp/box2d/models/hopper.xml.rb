common = { friction: 0.9, density: 1, group: -1 }

box2d {
  world(timestep: 0.02) {
    body(name: :torso, type: :dynamic, position: [0, 1.25]) {
      capsule(common.merge(from: [0, 0.2], to: [0, -0.2], radius: 0.05))
    }
    body(name: :thigh, type: :dynamic, position: [0, 0.825]) {
      capsule(common.merge(from: [0, 0.205], to: [0, -0.205], radius: 0.05))
    }
    body(name: :leg, type: :dynamic, position: [0, 0.35]) {
      capsule(common.merge(from: [0, 0.25], to: [0, -0.25], radius: 0.04))
    }
    body(name: :foot, type: :dynamic, position: [0.065, 0.1]) {
      capsule(common.merge(from: [-0.195, 0], to: [0.195, 0], radius: 0.06, friction: 2.0))
    }
    body(name: :ground, type: :static, position: [0, 0]) {
      fixture(shape: :polygon, box: [100, 0.05], friction: 2.0, density: 1, group: -2)
    }
    joint(
      type: :revolute,
      name: :thigh_joint,
      bodyA: :torso,
      bodyB: :thigh,
      anchor: [0, 1.05],
      limit: [-150.deg, 0.deg],
    )
    joint(
      type: :revolute,
      name: :leg_joint,
      bodyA: :thigh,
      bodyB: :leg,
      anchor: [0, 0.6],
      limit: [-150.deg, 0.deg],
    )
    joint(
      type: :revolute,
      name: :foot_joint,
      bodyA: :leg,
      bodyB: :foot,
      anchor: [0, 0.1],
      limit: [-45.deg, 45.deg],
    )
    state type: :xpos, com: [:torso, :thigh, :leg, :foot]
    state type: :ypos, com: [:torso, :thigh, :leg, :foot]
    state type: :apos, joint: :thigh_joint
    state type: :apos, joint: :leg_joint
    state type: :apos, joint: :foot_joint
    state type: :xvel, com: [:torso, :thigh, :leg, :foot]
    state type: :yvel, com: [:torso, :thigh, :leg, :foot]
    state type: :avel, joint: :thigh_joint
    state type: :avel, joint: :leg_joint
    state type: :avel, joint: :foot_joint
    max_torque = (2).Nm
    control(
      type: :torque,
      joint: :thigh_joint,
      ctrllimit: [-max_torque, max_torque]
    )
    control(
      type: :torque,
      joint: :leg_joint,
      ctrllimit: [-max_torque, max_torque]
    )
    control(
      type: :torque,
      joint: :foot_joint,
      ctrllimit: [-max_torque, max_torque]
    )
  }
}
