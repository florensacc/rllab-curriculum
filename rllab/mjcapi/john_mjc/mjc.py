"""
Wraps old version of mujoco simulator
This file is a nonsensical mess


"""

from .config import floatX,CTRL_ROOT,MJC_DATA_DIR
from .base_mdp import MDP
import mjcpy
import numpy as np, os.path as osp, h5py
from numpy import nan,inf,pi #pylint: disable=E0611,W0611
import theano, theano.tensor as TT #pylint: disable=F0401

MJC_BIG = 500

ST_UnbInvPos = 0     # Unbounded position, which cost & dynamics is invariant to
ST_UnbPos = 1        # Unbounded position
ST_BndPos = 2        # Bounded Position
ST_Height = 3        # Height of root link
ST_UnlimAng = 4      # Angle of continuous joint. (Can normalize to (-pi,pi))
ST_LimAng = 5        # Angle with upper and lower limits
ST_LinVel = 6        # Linear velocity
ST_AngVel = 7        # Angular velocity
ST_CtrlReal = 8

def arrayf(x):
    return np.array(x,floatX)
def false_vector(x):
    """return tensor with last dimension removed, with all entries = False"""
    return TT.take(x,0,axis=x.ndim-1) > np.inf


MJC_METADATA = {
    "3swimmer":{
        "actuated_joints":[3,4],
        "contact_type":mjcpy.ContactType.SPRING.real,
        "ctrl_bounds":arrayf([(-50,50)]*2).T,
        "state_desc": np.r_[ST_UnbInvPos, ST_Height, [ST_UnlimAng]*3, [ST_LinVel]*2, [ST_AngVel]*3],
        "default_state":arrayf([0,0,pi]+ [0]*7),
    },
    "hopper4ball":{
        "actuated_joints":[3,4,5],
        "contact_type":mjcpy.ContactType.SLCP.real,
        "ctrl_bounds":arrayf([(-15,15),(-60,60), (-70,70)]).T,
        "state_desc": np.r_[ST_UnbInvPos, ST_Height, [ST_UnlimAng]*4, [ST_LinVel]*2, [ST_AngVel]*4],        
        "state_bounds" : arrayf([(-MJC_BIG, MJC_BIG), (0.8, 2.0), (-1.0, 1.0)] + [(-MJC_BIG, MJC_BIG)]*9).T,
        "default_state" : arrayf([6.2420, 1.3086,0.2283, -0.0590,-0.4890,-0.1317,1.0986,0.4848,-0.4735,-0.3697,2.7385,-0.5730]),
        "hidx":1,
        "move_to_origin" : [0],
    },
    "walker2d":{
        "actuated_joints":[3,4,5,6,7,8],
        "contact_type":mjcpy.ContactType.SLCP.real,
        "ctrl_bounds": arrayf([[-149.765,  -62.8  ,  -48.932, -174.121,  -55.733,  -46.422],[ 174.942,   90.502,   93.728,  179.281,   85.147,   92.963]]),
        "state_desc": np.r_[ST_UnbInvPos, ST_Height, ST_UnlimAng, [ST_LimAng]*6, [ST_LinVel]*2, [ST_AngVel]*7],
        "timestep" : .01,
        "state_bounds" : arrayf([(-MJC_BIG, MJC_BIG), (0.8, 2.0), (-1.0, 1.0)] + [(-MJC_BIG,MJC_BIG)]*15).T,
        "default_state" : arrayf([0,1.2]+[0]*16),
        "hidx":1,
        "move_to_origin" : [0],
    },
    "tripod":{
        "actuated_joints":range(6,14),
        "contact_type":mjcpy.ContactType.SLCP.real,
        "ctrl_bounds":arrayf([(-100,100)]*8).T,
        "default_state":np.zeros(28,floatX),
        "hidx":2,
        "state_bounds":arrayf([(-MJC_BIG,MJC_BIG)]*2 + [(.75,1.5)] + [(-1,1)]*3 + [(-MJC_BIG,MJC_BIG)]*22).T,
        "move_to_origin" : [0,1],
    },
    "human3d":{
        "actuated_joints":  [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22],
        "contact_type":mjcpy.ContactType.SLCP.real,
        "ctrl_bounds":arrayf([(-100,100)]*19).T, # Questionable
        "timestep" : .01,
        "default_state": arrayf([0,0,0,1,0,0,0] + [0]*19 + [0]*25),
        "state_bounds" : arrayf([(-MJC_BIG, MJC_BIG)]*2 + [(1.1, 2.0)] + [(-MJC_BIG,MJC_BIG)]*48).T,
        "hidx":2,
        "sample_halfrange":.05,
        "move_to_origin" : [0,1],
    },
    "bvhmodel":{
        "actuated_joints":[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
        "contact_type":mjcpy.ContactType.SLCP.real,
        "ctrl_bounds": arrayf([[ -44.825, -151.393,  -51.322,   -4.245,  -35.99 ,   -2.49 ,  -17.991,   -7.453,  -17.849,   -3.452,  -17.403,  -67.772, -109.481,  -44.522, -278.725,  -36.595,   -6.911, -114.402,  -98.22 ,
        -21.511, -283.061,  -35.442,   -5.013],[  61.049,   46.269,   40.438,    9.106,   21.262,    3.475,    1.933,   10.576,   17.07 ,    6.278,    4.173,  235.061,  182.801,   32.303,   78.093,  143.632,   16.953,   64.873,  186.381,
         41.022,   85.903,  124.02 ,    3.768]]),
        "state_bounds" : arrayf( [(-MJC_BIG, MJC_BIG)]*2 + [(.70, 1.5)] + [(-MJC_BIG,MJC_BIG)]*60 ).T,
        # "default_state": arrayf([0,0,0,1,0,0,0] + [0]*(34-7+29)), #nqpos = 34, ndof=29
        # "default_state": arrayf([ 2.15 , -0.04 ,  0.923, -0.986,  0.048, -0.153, -0.05 ,  0.108, -0.099, -0.198,  0.922,  0.298, -0.169, -0.18 , -1.097,  0.928, -0.152,  0.32 ,  0.115, -1.583,  0.988,  0.054,  0.023,  0.14 ,
        # 0.262,  0.4  , -0.142,  0.931,  0.038, -0.361, -0.032,  0.276, -0.097, -0.101,  2.619, -0.071,  0.306,  1.595,  0.76 , -0.141, -1.592, -1.199, -1.123, -0.184,  0.043, -2.169,  1.393, -0.24 ,
        # 0.808, -0.269, -0.034, -0.292,  1.583,  0.428,  0.31 ,  2.838,  1.44 , -1.176, -0.372, -0.027, -8.661, -0.788,  2.474]),
        "default_state":arrayf([  9.306e-01,  -3.814e-04,   9.467e-01,  -9.935e-01,  -5.560e-02,  -9.874e-02,   1.072e-02,  -1.012e-01,  -4.076e-03,   1.814e-01,   9.416e-01,   1.977e-01,   1.849e-01,  -2.001e-01,
        -1.276e+00,   9.685e-01,  -2.375e-01,   6.598e-03,   7.401e-02,  -1.394e+00,   9.679e-01,  -5.499e-02,  -2.440e-01,   2.340e-02,   1.458e+00,   9.639e-02,   1.292e-01,   9.946e-01,
        -1.623e-02,  -8.944e-02,  -5.011e-02,   6.691e-01,  -3.023e-01,  -1.226e-01,   2.544e+00,   1.098e-01,   9.227e-01,   2.018e+00,   1.446e+00,  -1.688e-02,  -2.154e+00,  -1.713e+00,
         2.081e+00,  -1.672e+00,   3.104e+00,   1.246e+00,   4.895e-01,  -6.054e-01,  -9.580e-01,  -1.297e-01,   2.176e+00,  -2.964e+00,  -3.399e+00,   4.114e-01,  -2.996e+00,  -7.376e-01,
        -8.394e-01,  -2.569e+00,   2.371e+00,  -2.337e-02,  -3.762e+00,   4.790e+00,   5.178e-01])*0,
        "hidx":2,
        "timestep":.01,
        "sample_halfrange":0,
        "move_to_origin" : [0,1],
    },
}

def test_validate_metadata():
    for (mjcbasename,md) in MJC_METADATA.items():
        mdp = get_mjc_mdp_class(mjcbasename)
        assert (np.array(md["actuated_joints"]) < mdp.world_info["njnt"]).all()
        assert md["ctrl_bounds"].shape == (2,mdp.ctrl_dim())
        assert md["default_state"].size == mdp.state_dim()

def get_mjc_mdp_class(mjcbasename, kws=None):
    if kws is None: kws = {}
    name2cls = {
        "3swimmer":Swimming,
        "hopper4ball":LeggedLocomotion,
        "walker2d":LeggedLocomotion,
        "human3d":LeggedLocomotion,
        "bvhmodel":LeggedLocomotion,
        "tripod":LeggedLocomotion,
    }
    return name2cls[mjcbasename](mjcbasename, **kws)


def get_mjc_metadata(mjcbasename):
    return MJC_METADATA[mjcbasename]

class MJCMDP(MDP):
    def __init__(self,mjcbasename, ddp_friendly=False, frame_skip=1):
        self.ddp_friendly = ddp_friendly
        self.frame_skip=frame_skip


        self.md = get_mjc_metadata(mjcbasename)
        self.mjcbasename = mjcbasename
        self._setup_world()

    def _setup_world(self):
        mjcfile = osp.join(MJC_DATA_DIR, self.mjcbasename) + ".bin"
        self.world = mjcpy.MJCWorld(mjcfile)
        self.world_info = self.world.GetModel()
        # print self.world_info
        actuated_joints = self.md["actuated_joints"]
        self.actuated_dims = np.array([dofidx for (dofidx,jntidx) in enumerate(self.world_info["dof_jnt_id"]) if jntidx in actuated_joints])
        self.world.SetActuatedDims(self.actuated_dims)
        self.world.SetContactType(mjcpy.ContactType.values[self.md["contact_type"]])
        if "timestep" in self.md: 
            print "setting timestep to ",self.md["timestep"]
            self.world.SetTimestep(self.md["timestep"])
        import atexit
        def delworld(*_):
            try:
                del self.world
            except Exception: #pylint: disable=W0703
                pass
        atexit.register(delworld)

    def call(self,input_arrs):
        raise NotImplementedError

    def output_names(self):
        return ["x","o","c"]

    def initialize_mdp_arrays(self):
        x0 = self.default_state()
        halfrange = self.md.get("sample_halfrange", 0.2)
        x = np.random.uniform(low=x0-halfrange, high=x0+halfrange).astype(floatX).reshape(1,-1)

        return {
            "x" : x,
            "o" : np.zeros((1,self.obs_dim()),floatX),
        }

    def input_info(self):
        return {
            "x" : (self.state_dim(),floatX),
            "u" : (self.ctrl_dim(),floatX),
            "c" : (self.num_costs(),floatX)
        }    

    def output_info(self):
        return {
            "x" : (self.state_dim(),floatX),
            "o" : (self.obs_dim(),floatX),
            "c" : (self.num_costs(),floatX),
            "done" : (None,'uint8')
        }

    def plot(self, input_arrs):
        x = input_arrs["x"]
        assert x.shape[0]==1
        self.world.Plot(x[0].astype('float64'))

    def state_bounds(self):
        if "state_bounds" in self.md:
            return self.md["state_bounds"]
        else:
            raise RuntimeError

    def state_ranges(self):
        hdf = get_example_hdf(self.mjcbasename)
        xs = hdf["xs"].value
        return arrayf([xs.min(axis=0), xs.max(axis=0)])

    def state_desc(self):
        return self.md["state_desc"]

    def state_dim(self):
        return self.world_info["nqpos"] + self.world_info["ndof"]

    def ctrl_desc(self):
        return np.array([ST_CtrlReal]*self.ctrl_dim())

    def ctrl_bounds(self):
        bounds = arrayf(self.md["ctrl_bounds"])        
        if self.ddp_friendly:
            bounds[0] = -10000
            bounds[1] = 10000
        return bounds

    def ctrl_dim(self):
        return len(self.actuated_dims)

    def ctrl_dtype(self):
        return floatX

    def obs_dim(self):
        raise NotImplementedError

    def obs_ranges(self):
        return self.state_ranges()[:,1:]

    def obs_desc(self):
        return self.state_desc()[1:]

    def default_state(self):
        return self.md["default_state"]


    def __getstate__(self):
        print "pickling MJCMDP"
        d = self.__dict__.copy()
        del d["world"]
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self._setup_world()

    def vel_index(self):
        return self.world_info["nqpos"]


def logbarrier(z,t):
    """x >= 0"""
    return TT.switch(z > 0, -t*TT.log(z), np.array(np.inf, floatX))
def boundlogbarrier(x,bounds,t):
    lo,hi=bounds
    if x.ndim ==2:
        lo = lo[None,:]
        hi = hi[None,:]
    return logbarrier(x-lo,t) + logbarrier(hi-x,t) + 2*t*np.log( (hi-lo)/2 )

class Swimming(MJCMDP):
    def __init__(self, mjcbasename, frame_skip=1, ctrl_cost_coeff=1e-8, lin_vel_coeff=1.0):
        print "lalala"
        MJCMDP.__init__(self, mjcbasename, frame_skip=frame_skip)
        x = TT.matrix('x')
        u = TT.matrix('u')
        self.ctrl_cost_coeff = ctrl_cost_coeff
        self.lin_vel_coeff = lin_vel_coeff
        self._call = theano.function([x,u],self.symbolic_call(x,u))

    def dynamics(self,x,u):
        fmatrix = TT.matrix(dtype=floatX).type
        ctrl_lo, ctrl_hi = self.ctrl_bounds()
        @theano.as_op(itypes=[fmatrix,fmatrix],otypes=[fmatrix])
        def step(x_nd,u_ne):
            assert isinstance(x_nd,np.ndarray)
            u_ne = np.clip(u_ne, ctrl_lo, ctrl_hi)
            for _ in xrange(self.frame_skip): x_nd = self.world.StepMulti(np.ascontiguousarray(x_nd,dtype='float64'),np.ascontiguousarray(u_ne,dtype='float64')).astype(floatX)
            return x_nd
        return step(x,u)      

    def cost_rates(self, x,u):
        v = TT.take(x,5,axis=1)
        return [-self.lin_vel_coeff*v, self.ctrl_cost_coeff*(u**2).sum(axis=1)]

    def cost_names(self):
        return ["speed","ctrl"]

    def trial_done(self,x):
        return false_vector(x)        

    def symbolic_call(self, x, u):
        y = self.dynamics(x,u)
        o = y[:,1:]
        cs = self.cost_rates(x,u)
        c = TT.stack(*cs).T #pylint: disable=E1103
        d = self.trial_done(x)
        return [y,o,c,d]

    def call(self, input_arrs):
        x = input_arrs['x']
        u = input_arrs['u']
        y,o,c,d = self._call(x,u)
        return {"x":y,"o":o,"c":c,"done":d}

    def obs_dim(self):
        return self.state_dim()-1


class LeggedLocomotion(MJCMDP):
    def __init__(self, mjcbasename, vel_cost_coeff=1., vel_cost_type='linear', vel_cost_target=None, 
        ctrl_cost_coeff=2e-5, impact_cost_coeff=.01, done_cost_coeff=10., jntpos_cost_coeff = 0.,
        jntpos_use_l1 = False, jntpos_root_only = False,
        ddp_friendly=False, dof_armature=None, 
        use_kinematic_features=True, clip_impact_cost=False, frame_skip=1):
        

        # First set dof_armature since it's needed in _setup_world
        self.dof_armature = dof_armature    

        MJCMDP.__init__(self, mjcbasename,ddp_friendly,frame_skip=frame_skip)


        self.vel_cost_coeff = vel_cost_coeff
        self.vel_cost_type = vel_cost_type
        self.vel_cost_target = vel_cost_target
        self.ctrl_cost_coeff = ctrl_cost_coeff
        self.impact_cost_coeff = impact_cost_coeff
        self.done_cost_coeff = done_cost_coeff
        self.jntpos_cost_coeff = jntpos_cost_coeff
        self.jntpos_use_l1 = jntpos_use_l1
        self.jntpos_root_only = jntpos_root_only


        self.use_kinematic_features = use_kinematic_features
        self.clip_impact_cost = clip_impact_cost

        assert self.vel_cost_type in ("linear","quadratic")
        self._obs_dim = None

        x = TT.matrix('x')
        u = TT.matrix('u')
        self._call = theano.function([x,u],self.symbolic_call(x,u))
        oa = self.call({"x":np.zeros((1,self.state_dim()),floatX), "u":np.zeros((1,self.ctrl_dim()),floatX)})
        self._obs_dim = oa["o"].size


    def _setup_world(self):
        MJCMDP._setup_world(self)
        if self.dof_armature is not None:             
            print "modifying armature"
            armature_arr = np.ones(self.world_info["ndof"])*self.dof_armature
            isroot_joint = self.world_info["jnt_body_id"]==1
            rootbody_dofs = np.array([dofidx for (dofidx,jntidx) in enumerate(self.world_info["dof_jnt_id"]) if isroot_joint[jntidx]])
            armature_arr[rootbody_dofs] = 0
            self.world.SetModel(dict(dof_armature=armature_arr))

    def dynamics_costs_obs(self,x,u):
        fmatrix = TT.matrix(dtype=floatX).type
        uvector = TT.vector(dtype='int8').type
        ctrl_lo, ctrl_hi = self.ctrl_bounds()


        @theano.as_op(itypes=[fmatrix,fmatrix,uvector],otypes=[fmatrix,fmatrix,fmatrix,fmatrix,fmatrix])
        def stepmulti2op(x_nd,u_ne,done_n):
            x_nd = x_nd.copy()
            u_ne = np.clip(u_ne, ctrl_lo, ctrl_hi)
            move_to_origin = self.md["move_to_origin"]
            offset_n2 = x_nd[:,move_to_origin].copy()
            x_nd[:,move_to_origin] -= offset_n2
            x_nd,f,dcom,dist,kin = self.world.StepMulti2(x_nd.astype("float64"),u_ne.astype("float64"),done_n)
            for _ in xrange(self.frame_skip-1):
                x_nd,f1,dcom1,dist,kin = self.world.StepMulti2(x_nd.astype("float64"),u_ne.astype("float64"),done_n)
                dcom += dcom1
                f += f1
            f /= self.frame_skip

            dist = np.clip(dist, 0, .1) # XXX clip level ad hoc 
            # Consider using nan_to_num here
            x_nd[:,move_to_origin] += offset_n2
            return (x_nd.astype(floatX),f.astype(floatX),dcom.astype(floatX),dist.astype(floatX),kin.astype(floatX))

        done = self.trial_done(x)
        notdone = 1 - done

        y,f,dcom,dist,kin = stepmulti2op(x,u,done)


        if self.vel_cost_type == "linear":
            cost_vel = (-self.vel_cost_coeff/self.world_info["timestep"]) * dcom[:,0]
        elif self.vel_cost_type == "quadratic":
            cost_vel = TT.square(dcom[:,0]/self.world_info["timestep"] - self.vel_cost_target) #pylint: disable=E1111
        else:
            raise ValueError
        cost_ctrl = .5*self.ctrl_cost_coeff*TT.square(u).sum(axis=1)
        cost_impact = .5*self.impact_cost_coeff * TT.square(f).sum(axis=1)
        if self.clip_impact_cost:
            cost_impact = TT.minimum(cost_impact, self.clip_impact_cost) #pylint: disable=E1111


        jntpos_mask = self.world_info["jnt_islimited"]
        if self.jntpos_root_only: jntpos_mask &= (self.world_info["jnt_body_id"]==1)
        jntpos_inds = np.flatnonzero(jntpos_mask)
        jntpos_dofs = np.array([dofidx for (dofidx,jntidx) in enumerate(self.world_info["dof_jnt_id"]) if jntidx in jntpos_inds])        
        cost_jntpos = (.5*self.jntpos_cost_coeff) * (TT.abs_ if self.jntpos_use_l1 else TT.square)(y[:,jntpos_dofs]).sum(axis=1)

        cost_done = (done-1)*self.done_cost_coeff
        feats = [y[:,1:],f,dist]
        if self.use_kinematic_features: feats.append(kin)
        obs = TT.concatenate(feats,axis=1)
        return [TT.switch(done[:,None], x, y),  [notdone*cost_vel, notdone*cost_ctrl, notdone*cost_impact, notdone*cost_jntpos, cost_done] , obs ]

    def cost_names(self):
        return ["vel","ctrl","impact","jntpos","done"] 

    def trial_done(self, x):
        if self.ddp_friendly or self.md.get("no_termination",False):
            return false_vector(x)
        else:
            mins,maxes = self.state_bounds()
            # return ((x>maxes)|(x<mins)).any(axis=x.ndim-1)

            return 1-((x<maxes) * (x>mins)).all(axis=1)
            # return ( TT.neg( TT.and_(x<maxes,x>mins)) ).any(axis=1)

    def default_state(self):
        return self._project_to_above_ground(self.md["default_state"])
    
    def _project_to_above_ground(self,x):
        # XXX only makes sense on flat ground
        dists = self.world.ComputeContacts(x.astype('float64'))
        out = x.copy()
        hidx = self.md["hidx"]
        out[hidx] -= min(dists.min(),0)
        return out

    def obs_ranges(self):
        raise NotImplementedError

    def initialize_mdp_arrays(self):
        arrs = MJCMDP.initialize_mdp_arrays(self)
        arrs["x"] = self._project_to_above_ground(arrs["x"][0]).reshape(1,-1)
        return arrs

    def symbolic_call(self, x, u):
        y,cs,o = self.dynamics_costs_obs(x,u)
        c = TT.stack(*cs).T #pylint: disable=E1103
        d = self.trial_done(x)
        return [y,o,c,d]

    def call(self, input_arrs):
        x = input_arrs['x']
        u = input_arrs['u']
        y,o,c,d = self._call(x,u)
        bad_n = np.zeros(y.shape[0],'bool')        
        for arr in (y,o,c):
            bad_n |= ~np.isfinite(arr).all(axis=1)
        if bad_n.any():
            print "ERROR: rollouts had invalid values in y: %i, o: %i, c: %i"%( (~np.isfinite(y).all(axis=1)).sum(),(~np.isfinite(c).all(axis=1)).sum(),(~np.isfinite(o).all(axis=1)).sum())
            d[bad_n] = 1
            o[bad_n] = 0
            c[bad_n] = 0
            y[bad_n] = 0
        return {"x":y,"o":o,"c":c,"done":d}

    def obs_dim(self):
        return self._obs_dim


def get_example_hdf(mjcbasename):
    fname = {
    "hopper4ball":"ex_hopper.h5", 
    "walker2d":"ex_walker_slcp.h5",
    "3swimmer":"ex_swimmer.h5","bvhmodel":"ex_bvhmodel_projected.h5",
    "tripod":"tripod_random_example_armature0.25.h5",
    "human3d":"human3d_random_example_armature0.1.h5",
    }[mjcbasename]
    hdf = h5py.File(osp.join(CTRL_ROOT,"domain_data/mjc_examples",fname), "r")
    return hdf    



def main():
    for mjcbasename in MJC_METADATA.iterkeys():
        mdp = get_mjc_mdp_class(mjcbasename)
        mdp.validate()

    test_validate_metadata()

if __name__ == "__main__":
    main()
