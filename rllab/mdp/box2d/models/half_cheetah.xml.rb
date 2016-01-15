common = { friction: 0.9, density: 1, group: -1, radius: 0.046 }
foot_friction = 200.0
data = {}
box2d {
  world(timestep: 0.001) {
    base(body: {position: [0, 0.67]}) {
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
        len, ang = 0.145*2, -1.046666667 #-0.523333333#-3.8 + 3.14
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
          len, ang = 0.15*2, -2.616666667#-2.03
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
            len, ang = 0.094*2, -1.308333333#-0.27
            l = [0, 0]
            r = [
                 l[0]+len*Math.cos(ang),
                 l[1]+len*Math.sin(ang)
                ]
            data[:bfoot_end] = r
            capsule(common.merge(from: l, to: r, friction: foot_friction))
          }
        }
      }
      base(body: {position: [0.5, 0]}) {
        len, ang = 0.133*2, -1.993333333#0.52
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
          len, ang = 0.106*2, -1.1
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
            len, ang = 0.07*2, -1.1
            l = [0, 0]
            r = [
                 l[0]+len*Math.cos(ang),
                 l[1]+len*Math.sin(ang)
                ]
            data[:ffoot_end] = r
            capsule(common.merge(from: l, to: r, friction: foot_friction))
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
          limit: [-60.deg, 30.deg],
          )
    control(
      type: :torque,
      joint: :bthigh_joint,
      ctrllimit: [-120.Nm, 120.Nm]
    )
    joint(
          type: :revolute,
          name: :bshin_joint,
          bodyA: :bthigh,
          bodyB: :bshin,
          anchor: data[:bshin_anchor],
          limit: [-45.deg, 45.deg],
          )
    control(
      type: :torque,
      joint: :bshin_joint,
      ctrllimit: [-90.Nm, 90.Nm]
    )
    joint(
          type: :revolute,
          name: :bfoot_joint,
          bodyA: :bshin,
          bodyB: :bfoot,
          anchor: data[:bfoot_anchor],
          limit: [-75.deg, 45.deg],
          )
    control(
      type: :torque,
      joint: :bfoot_joint,
      ctrllimit: [-60.Nm, 60.Nm]
    )
    joint(
          type: :revolute,
          name: :fthigh_joint,
          bodyA: :torso,
          bodyB: :fthigh,
          anchor: data[:fthigh_anchor],
          limit: [-60.deg, 20.deg],
          )
    control(
      type: :torque,
      joint: :fthigh_joint,
      ctrllimit: [-120.Nm, 120.Nm]
    )
    joint(
          type: :revolute,
          name: :fshin_joint,
          bodyA: :fthigh,
          bodyB: :fshin,
          anchor: data[:fshin_anchor],
          limit: [-110.deg, 50.deg],
          )
    control(
      type: :torque,
      joint: :fshin_joint,
      ctrllimit: [-60.Nm, 60.Nm]
    )
    joint(
          type: :revolute,
          name: :ffoot_joint,
          bodyA: :fshin,
          bodyB: :ffoot,
          anchor: data[:ffoot_anchor],
          limit: [-120.deg, 20.deg],
          )
    control(
      type: :torque,
      joint: :ffoot_joint,
      ctrllimit: [-30.Nm, 30.Nm]
    )
    body(name: :ground, type: :static, position: [0, 0]) {
      fixture(shape: :polygon, box: [100, 0.05], friction: 2.0, density: 1, group: -2)
    }
    state type: :ypos, body: :bfoot, local: data[:bfoot_end]
    state type: :yvel, body: :bfoot, local: data[:bfoot_end]
    # indicator state neglected
    state type: :ypos, body: :torso
    state type: :xvel, body: :torso
    state type: :yvel, body: :bthigh
    state type: :yvel, body: :fthigh
    state type: :ypos, body: :ffoot, local: data[:ffoot_end]
    state type: :yvel, body: :ffoot, local: data[:ffoot_end]
    [
     :bthigh_joint, :bshin_joint, :bfoot_joint,
     :fthigh_joint, :fshin_joint, :ffoot_joint,
    ].each do |joint|
      state type: :apos, joint: joint
      state type: :avel, joint: joint
    end
    state type: :apos, joint: :bthigh_joint
    state type: :apos, joint: :bshin_joint
    state type: :apos, joint: :bfoot_joint

  }
}
