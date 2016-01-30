link_len = 1
link_mass = 0.01
link_width = 0.1

link_track_group = -1

cart_width = 4.0 / (12 ** 0.5)
cart_height = 3.0 / (12 ** 0.5)
cart_friction = 0.0005

box2d {
  world(timestep: 0.001, velitr: 200, positr: 200) {
    body(name: :cart, type: :dynamic, position: [0, 0]) {
      rect(
        box: [cart_width / 2, cart_height / 2],
        density: 1,
        friction: cart_friction,
        group: link_track_group,
      )
    }
    body(name: :link1, type: :dynamic, position: [0, cart_height/2]) {
      rect(
        from: [0, 0], to: [0, link_len],
        radius: link_width / 2,
        density: link_mass / link_len / link_width,
        group: link_track_group
      )
      # fixture(
      #         shape: :circle, center: [0, link_len], radius: link_width/2, density: 10,
      #   group: link_track_group
      # )
    }
    body(name: :link2, type: :dynamic, position: [0, cart_height/2 + link_len]) {
      rect(
        from: [0, 0], to: [0, link_len],
        radius: link_width / 2,
        density: link_mass / link_len / link_width,
        group: link_track_group
      )
      # fixture(
      #         shape: :circle, center: [0, link_len], radius: link_width/2, density: 10,
      #   group: link_track_group
      # )
    }
    body(name: :track, type: :static, position: [0, 0]) {
      rect(
        box: [100, 0.1],
        group: link_track_group
      )
    }
    joint(
      type: :prismatic,
      name: :track_cart,
      bodyA: :track,
      bodyB: :cart,
    )
    joint(
      type: :revolute,
      name: :link_joint_1,
      bodyA: :cart,
      bodyB: :link1,
      localAnchorA: [0, cart_height/2],
      localAnchorB: [0, 0],
    )
    joint(
      type: :revolute,
      name: :link_joint_2,
      bodyA: :link1,
      bodyB: :link2,
      localAnchorA: [0, link_len],
      localAnchorB: [0, 0],
    )
    control(
      type: :force,
      body: :cart,
      anchor: [0, 0],
      direction: [1, 0],
      ctrllimit: [-10.N, 10.N],
    )

    state type: :xpos, body: :cart
    state type: :xvel, body: :cart
    state type: :apos, body: :link1, transform: :sin
    state type: :apos, body: :link1, transform: :cos
    state type: :avel, body: :link1
    state type: :apos, body: :link2, transform: :sin
    state type: :apos, body: :link2, transform: :cos
    state type: :avel, body: :link2
  }
}
