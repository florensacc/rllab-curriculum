<mujoco model="swimmer">
  <compiler inertiafromgeom="true" angle="degree" coordinate="local" />
  <custom>
  <numeric name="frame_skip" data="1" />
  </custom>
  <option timestep="0.05" density="1000" viscosity="0.1" collision="predefined" integrator="Euler" iterations="10" >
    <flag warmstart="disable" />
  </option>
  <default>
    <geom contype='1' conaffinity='1' condim='1' rgba='0.8 0.6 .4 1' 
      material="geom"/>
    <!--<joint armature='1'  />-->
  </default>
  <asset>
    <texture type="skybox" builtin="gradient" width="100" height="100" rgb1="1 1 1" rgb2="0 0 0" />
    <texture name="texgeom" type="cube" builtin="flat" mark="cross" width="127" height="1278" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" markrgb="1 1 1" random="0.01" />
    <texture name="texplane" type="2d" builtin="checker" rgb1="0 0 0" rgb2="0.8 0.8 0.8" width="100" height="100" />
    <material name='MatPlane' texture="texplane" shininess="1" texscale="0.3 0.3" specular="1"  reflectance="0.5" />
    <material name='geom' texture="texgeom" texuniform="true" />
  </asset>
	<worldbody>
    <light directional="true" cutoff="100" exponent="1" diffuse="1 1 1" specular=".1 .1 .1" pos="0 0 1.3" dir="-0 0 -1.3" />
    <geom name='floor' material="MatPlane" pos='0 0 -0.1' size='40 40 0.1' type='plane' conaffinity='1' rgba='0.8 0.9 0.8 1' condim='3' />
    <geom name='origin' pos='0 0 0' size='0.1 0.1 0.1' type='box' />
		<!--  =================   MAZE  ================= /-->

    <!--<geom type="box" pos="0 0 0" size=".1 1 1" />-->

    <%namespace file="utils.mako" name="utils" />

    <%
        height = 0.5
        size_scaling = 4
        structure = [
            [1, 1, 1, 1, 1],
            [1, 'r', 0, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
    %>
    ${utils.make_maze(structure, height=height, size_scaling=size_scaling)}

		<!--  ================= SWIMMER ================= /-->

    <body name="torso" pos="${utils.find_robot(structure, size_scaling=size_scaling)}">
    <geom type="capsule" fromto="1.5 0 0 0.5 0 0" size="0.1" density="1000" />
          <joint pos="0 0 0" type="slide" name="slider1" axis="1 0 0" />
          <joint pos="0 0 0" type="slide" name="slider2" axis="0 1 0" />
          <joint name="rot" type="hinge" pos="0 0 0" axis="0 0 1" />

          <body name="mid" pos="0.5 0 0">
            <geom type="capsule" fromto="0 0 0 -1 0 0" size="0.1" density="1000" />
            <joint name="rot2" type="hinge" pos="0 0 0" axis="0 0 1" range="-100 100" limited="true" />
            <body name="back" pos="-1 0 0">
              <geom type="capsule" fromto="0 0 0 -1 0 0" size="0.1" density="1000" />
              <joint name="rot3" type="hinge" pos="0 0 0" axis="0 0 1" range="-100 100" limited="true" />
            </body>
          </body>
        </body>	
	</worldbody>

  <actuator>
    <motor joint="rot2" ctrllimited="true" ctrlrange="-50 50"/>
    <motor joint="rot3" ctrllimited="true" ctrlrange="-50 50"/>
  </actuator>
</mujoco>
