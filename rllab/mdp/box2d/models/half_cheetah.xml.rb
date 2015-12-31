common = { friction: 0.9, density: 1, group: -1, radius: 0.046 }
data = {}
box2d {
  world(timestep: 0.01) {
    base(body: {position: [0, -0.7]}) {
      body(name: :torso, type: :dynamic) {
        torso_l, torso_r = [[-0.5, 0], [0.5, 0]]
        capsule(common.merge(from: torso_l, to: torso_r))

        len, ang = 0.15*2, 0.87
        l = torso_r
        r = [
             l[0]+len*Math.cos(ang),
             l[1]+len*Math.sin(ang)
            ]
        capsule(common.merge(from: l, to: r))
      }
      base(body: {position: [-0.5, 0]}) {
        len, ang = 0.145*2, -3.8 + 3.14
        l = [0, 0]
        data[:bthigh_anchor] = query(:body, :position)
        r = [
             l[0]+len*Math.cos(ang),
             l[1]+len*Math.sin(ang)
            ]
        body(name: :bthigh, type: :dynamic) {
          capsule(common.merge(from: l, to: r))
        }

        base(body: {position: r}) {
          len, ang = 0.15*2, -2.03
          l = [0, 0]
          data[:bshin_anchor] = query(:body, :position)
          r = [
               l[0]+len*Math.cos(ang),
               l[1]+len*Math.sin(ang)
              ]
          body(name: :bshin, type: :dynamic) {
            capsule(common.merge(from: l, to: r))
          }
          data[:bfoot_anchor] = query(:body, :position).base_merge(r)
          body(name: :bfoot, type: :dynamic, position: r) {
            len, ang = 0.094*2, -0.27
            l = [0, 0]
            r = [
                 l[0]+len*Math.cos(ang),
                 l[1]+len*Math.sin(ang)
                ]
            capsule(common.merge(from: l, to: r))
          }
        }
      }
      base(body: {position: [0.5, 0]}) {
        len, ang = 0.133*2, 0.52 + 3.14
        l = [0, 0]
        data[:fthigh_anchor] = query(:body, :position)
        r = [
             l[0]+len*Math.cos(ang),
             l[1]+len*Math.sin(ang)
            ]
        body(name: :fthigh, type: :dynamic) {
          capsule(common.merge(from: l, to: r))
        }

        base(body: {position: r}) {
          len, ang = 0.106*2, -0.6
          l = [0, 0]
          data[:fshin_anchor] = query(:body, :position)
          r = [
               l[0]+len*Math.cos(ang),
               l[1]+len*Math.sin(ang)
              ]
          body(name: :fshin, type: :dynamic) {
            capsule(common.merge(from: l, to: r))
          }
          data[:ffoot_anchor] = query(:body, :position).base_merge(r)
          body(name: :ffoot, type: :dynamic, position: r) {
            len, ang = 0.07*2, -0.6
            l = [0, 0]
            r = [
                 l[0]+len*Math.cos(ang),
                 l[1]+len*Math.sin(ang)
                ]
            capsule(common.merge(from: l, to: r))
          }
        }
      }
    }
    joint(
          type: :revolute,
          name: :bthigh_joint,
          bodyA: :torso,
          bodyB: :bthigh,
          anchor: data[:bthigh_anchor],
          limit: [-2, 2],
          )
    joint(
          type: :revolute,
          name: :bshin_joint,
          bodyA: :bthigh,
          bodyB: :bshin,
          anchor: data[:bshin_anchor],
          limit: [-2, 2],
          )
    joint(
          type: :revolute,
          name: :bfoot_joint,
          bodyA: :bshin,
          bodyB: :bfoot,
          anchor: data[:bfoot_anchor],
          limit: [-2, 2],
          )
    joint(
          type: :revolute,
          name: :bthigh_joint,
          bodyA: :torso,
          bodyB: :fthigh,
          anchor: data[:fthigh_anchor],
          limit: [-2, 2],
          )
    joint(
          type: :revolute,
          name: :bshin_joint,
          bodyA: :fthigh,
          bodyB: :fshin,
          anchor: data[:fshin_anchor],
          limit: [-2, 2],
          )
    joint(
          type: :revolute,
          name: :bfoot_joint,
          bodyA: :fshin,
          bodyB: :ffoot,
          anchor: data[:ffoot_anchor],
          limit: [-2, 2],
          )
    body(name: :ground, type: :static, position: [0, -2.0]) {
      fixture(shape: :polygon, box: [100, 0.05], friction: 2.0, density: 1, group: -2)
    }
  }
}
