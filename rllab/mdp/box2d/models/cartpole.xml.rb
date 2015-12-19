cart_width = 4.0 / (12 ** 0.5)
cart_height = 3.0 / (12 ** 0.5)

pole_width = 0.05
pole_height = 2.0

cart_friction = 0.0005
pole_friction = 0.000002

box2d {
  world(timestep: 0.05) {
    body(name: :cart, type: :dynamic, position: [0, cart_height / 2]) {
      rect(box: [cart_width / 2, cart_height / 2], density: 1, friction: cart_friction)
    }
    body(name: :pole, type: :dynamic, position: [0, cart_height]) {
      rect(from: [0, 0], to: [0, pole_height], radius: pole_width / 2, density: 1, friction: cart_friction)
    }
    body(name: :track, type: :static, position: [0, -0.1]) {
      rect(box: [100, 0.1], friction: pole_friction)
    }
    joint(
      type: :revolute,
      name: :pole_joint,
      bodyA: :cart,
      bodyB: :pole,
      anchor: [0, cart_height],
    )
    state type: :xpos, body: :cart
    state type: :xvel, body: :cart
    state type: :apos, body: :pole
    state type: :avel, body: :pole
    control(
      type: :force,
      body: :cart,
      anchor: [0, 0],
      direction: [1, 0],
      ctrllimit: [-10.N, 10.N],
    )
  }
}
