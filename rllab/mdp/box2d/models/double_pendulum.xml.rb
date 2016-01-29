link_len = 1
link_mass = 0.5
link_width = 0.1

link_track_group = -1

box2d {
  world(timestep: 0.05, velitr: 20, positr: 20) {
    body(name: :link1, type: :dynamic, position: [0, 0]) {
      rect(
        from: [0, 0], to: [0, -link_len],
        radius: link_width / 2,
        density: link_mass / link_len / link_width,
        group: link_track_group
      )
    }
    body(name: :link2, type: :dynamic, position: [0, -link_len]) {
      rect(
        from: [0, 0], to: [0, -link_len],
        radius: link_width / 2,
        density: link_mass / link_len / link_width,
        group: link_track_group
      )
    }
    body(name: :track, type: :static, position: [0, -0.1]) {
      rect(
        box: [100, 0.1],
        group: link_track_group
      )
    }
    joint(
      type: :revolute,
      name: :link_joint_1,
      bodyA: :track,
      bodyB: :link1,
      anchor: [0, 0],
    )
    joint(
      type: :revolute,
      name: :link_joint_2,
      bodyA: :link1,
      bodyB: :link2,
      anchor: [0, -link_len],
    )
    control(
      type: :torque,
      joint: :link_joint_1,
      ctrllimit: [-3.N, 3.N],
    )
    control(
      type: :torque,
      joint: :link_joint_2,
      ctrllimit: [-3.N, 3.N],
    )
    state type: :apos, body: :link1
    state type: :avel, body: :link1
    state type: :apos, body: :link2
    state type: :avel, body: :link2
  }
}
