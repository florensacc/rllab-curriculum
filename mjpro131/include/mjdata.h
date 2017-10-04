//-----------------------------------//
//  This file is part of MuJoCo.     //
//  Copyright (C) 2016 Roboti LLC.   //
//-----------------------------------//


#pragma once


//---------------------------- primitive types (mjt) ------------------------------------
    
typedef enum _mjtWarning            // warning types
{
    mjWARN_INERTIA      = 0,        // (near) singular inertia matrix
    mjWARN_CONTACTFULL,             // too many contacts in contact list
    mjWARN_CNSTRFULL,               // too many constraints
    mjWARN_VGEOMFULL,               // too many visual geoms
    mjWARN_BADQPOS,                 // bad number in qpos
    mjWARN_BADQVEL,                 // bad number in qvel
    mjWARN_BADQACC,                 // bad number in qacc
    mjWARN_SOLVER,					// poor solver convergence

    mjNWARNING                      // number of warnings
} mjtWarning;


typedef enum _mjtTimer
{
    // main api
    mjTIMER_STEP        = 0,        // step
    mjTIMER_INVERSE,                // inverse

    // breakdown of step
    mjTIMER_POSITION,               // fwdPosition
    mjTIMER_VELOCITY,               // fwdVelocity
    mjTIMER_ACTUATION,              // fwdActuation
    mjTIMER_ACCELERATION,           // fwdAcceleration
    mjTIMER_CONSTRAINT,             // fwdConstraint
    mjTIMER_SENSOR,                 // sensor
    mjTIMER_ENERGY,                 // energy

    // breakdown of fwdPosition
    mjTIMER_POS_KINEMATICS,         // kinematics, com, tendon, transmission
    mjTIMER_POS_INERTIA,            // inertia computations
    mjTIMER_POS_COLLISION,          // collision detection
    mjTIMER_POS_MAKE,               // make constraints
    mjTIMER_POS_PROJECT,            // project constraints

    mjNTIMER                        // number of timers
} mjtTimer;


//------------------------------ mjContact ----------------------------------------------

struct _mjContact                   // result of collision detection functions
{
    // contact parameters set by geom-specific collision detector
    mjtNum dist;                    // distance between nearest points; neg: penetration
    mjtNum pos[3];                  // position of contact point: midpoint between geoms
    mjtNum frame[9];                // normal is in [0-2]

    // contact parameters set by mj_collideGeoms
    mjtNum includemargin;           // include if dist<includemargin=margin-gap
    mjtNum friction[5];             // tangent1, 2, spin, roll1, 2
    mjtNum solref[mjNREF];          // constraint solver reference
    mjtNum solimp[mjNIMP];          // constraint solver impedance

    // storage used internally by constraint solver
    mjtNum mu;						// friction of regularized cone
    mjtNum coef[5];					// coefficients of middle-zone distance formula
    int zone;						// 0: top, 1: middle, 2: bottom

    // contact descriptors set by mj_collideGeoms
    int dim;                        // contact space dimensionality: 1, 3, 4 or 6
    int geom1;                      // id of geom 1
    int geom2;                      // id of geom 2

    // flag set by mj_fuseContact or mj_instantianteEquality
    int exclude;                    // 0: include, 1: in gap, 2: fused, 3: equality

    // address computed by mj_instantiateContact
    int efc_address;                // address in efc; -1: not included, -2-i: distance constraint i ???
};
typedef struct _mjContact mjContact;


//---------------------------------- mjData ---------------------------------------------

struct _mjData
{
    // constant sizes
    int nstack;                     // number of mjtNums that can fit in stack
    int nbuffer;                    // size of main buffer in bytes

    // stack info
    int pstack;                     // first available mjtNum address in stack
    int maxstackuse;                // keep track of maximum stack allocation

        // also keep track of maxefcuse and maxconuse ???

    // variable sizes
    int ne;                         // number of equality constraints
    int nf;                         // number of friction constraints
    int nefc;                       // number of constraints
    int ncon;                       // number of detected contacts
    int nwarning[mjNWARNING];       // how many times is each warning generated
    int warning_info[mjNWARNING];   // warning-related info (index or size)

    // timing
    mjtNum timer_duration[mjNTIMER];// accumulated duration in microsec
    mjtNum timer_ncall[mjNTIMER];   // number of calls over which duration accumulated
    mjtNum mocaptime[3];            // hardware time, cpu time, sim time

    // global properties
    mjtNum time;                    // simulation time
    mjtNum energy[2];               // kinetic, potential energy
    mjtNum solverstat[4];           // iterations, gradnorm, fwdinv_qfrc, fwdinv_efc
    mjtNum solvertrace[mjNTRACE];	// gradnorm over solver iterations (0: terminator)

    //-------------------------------- end of info header

    // buffers
    void*     buffer;               // main buffer; all pointers point in it    (nbuffer bytes)
    mjtNum*   stack;                // stack buffer                             (nstack mjtNums)

    //-------------------------------- main inputs and outputs of the computation

    // state
    mjtNum*   qpos;                 // position                                 (nq x 1)
    mjtNum*   qvel;                 // velocity                                 (nv x 1)
    mjtNum*   act;                  // actuator activation                      (na x 1)

    // control
    mjtNum*   ctrl;                 // control                                  (nu x 1)
    mjtNum*   qfrc_applied;         // applied generalized force                (nv x 1)
    mjtNum*   xfrc_applied;         // applied Cartesian force/torque           (nbody x 6)

    // dynamics
    mjtNum*   qacc;                 // acceleration                             (nv x 1)
    mjtNum*   act_dot;              // time-derivative of actuator activation   (na x 1)

    // mocap data
    mjtNum*  mocap_pos;             // positions of mocap bodies                (nmocap x 3)
    mjtNum*  mocap_quat;            // orientations of mocap bodies             (nmocap x 4)

    // user data
    mjtNum*  userdata;             // user data, not touched by engine          (nuserdata x 1)

    // sensors
    mjtNum*  sensordata;           // sensor data array                         (nsensordata x 1)

    //-------------------------------- POSITION dependent

    // computed by mj_fwdPosition/mj_kinematics
    mjtNum*   xpos;                 // Cartesian position of body frame         (nbody x 3)
    mjtNum*   xquat;                // Cartesian orientation of body frame      (nbody x 4)
    mjtNum*   xmat;                 // Cartesian orientation of body frame      (nbody x 9)
    mjtNum*   xipos;                // Cartesian position of body com           (nbody x 3)
    mjtNum*   ximat;                // Cartesian orientation of body inertia    (nbody x 9)
    mjtNum*   xanchor;              // Cartesian position of joint anchor       (njnt x 3)
    mjtNum*   xaxis;                // Cartesian joint axis                     (njnt x 3)
    mjtNum*   geom_xpos;            // Cartesian geom position                  (ngeom x 3)
    mjtNum*   geom_xmat;            // Cartesian geom orientation               (ngeom x 9)
    mjtNum*   site_xpos;            // Cartesian site position                  (nsite x 3)
    mjtNum*   site_xmat;            // Cartesian site orientation               (nsite x 9)
    mjtNum*   cam_xpos;             // Cartesian camera position                (ncam x 3)
    mjtNum*   cam_xmat;             // Cartesian camera orientation             (ncam x 9)
    mjtNum*   light_xpos;           // Cartesian light position                 (nlight x 3)
    mjtNum*   light_xdir;           // Cartesian light direction                (nlight x 3)

    // computed by mj_fwdPosition/mj_comPos
    mjtNum*   com_subtree;          // center of mass of each subtree           (nbody x 3)
    mjtNum*   cdof;                 // com-based motion axis of each dof        (nv x 6)
    mjtNum*   cinert;               // com-based body inertia and mass          (nbody x 10)

    // computed by mj_fwdPosition/mj_tendon
    int*      ten_wrapadr;          // start address of tendon's path           (ntendon x 1)
    int*      ten_wrapnum;          // number of wrap points in path            (ntendon x 1)
    mjtNum*   ten_length;           // tendon lengths                           (ntendon x 1)
    mjtNum*   ten_moment;           // tendon moment arms                       (ntendon x nv)
    int*      wrap_obj;             // geom id; -1: site; -2: pulley            (nwrap*2 x 1)
    mjtNum*   wrap_xpos;            // Cartesian 3D points in all path          (nwrap*2 x 3)

    // computed by mj_fwdPosition/mj_transmission
    mjtNum*   actuator_length;      // actuator lengths                         (nu x 1)
    mjtNum*   actuator_moment;      // actuator moment arms                     (nu x nv)

    // computed by mj_fwdPosition/mj_crb
    mjtNum*   crb;                  // com-based composite inertia and mass     (nbody x 10)
    mjtNum*   qM;                   // total inertia                            (nM x 1)

    // computed by mj_fwdPosition/mj_factorM
    mjtNum*   qLD;                  // L'*D*L factorization of M                (nM x 1)
    mjtNum*   qLDiagInv;            // 1/diag(D)                                (nv x 1)
    mjtNum*   qLDiagSqrtInv;        // 1/sqrt(diag(D))                          (nv x 1)

    // computed by mj_fwdPosition/mj_collision
    mjContact* contact;             // list of all detected contacts            (nconmax x 1)

    // computed by mj_fwdPosition/mj_makeConstraint
    int*      efc_type;             // constraint type (mjtConstraint)          (njmax x 1)
    int*      efc_id;               // id of object of specified type           (njmax x 1)
    int*      efc_rownnz;           // number of non-zeros in Jacobian row      (njmax x 1)
    int*      efc_rowadr;           // row start address in colind array        (njmax x 1)
    int*      efc_colind;           // column indices in sparse Jacobian        (njmax x nv)
    int*      efc_rownnz_T;         // number of non-zeros in Jacobian row  T   (nv x 1)
    int*      efc_rowadr_T;         // row start address in colind array    T   (nv x 1)
    int*      efc_colind_T;         // column indices in sparse Jacobian    T   (nv x njmax)
    mjtNum*   efc_solref;           // constraint solver reference              (njmax x mjNREF)
    mjtNum*   efc_solimp;           // constraint solver impedance              (njmax x mjNIMP)
    mjtNum*   efc_margin;           // inclusion margin (contact)               (njmax x 1)
    mjtNum*   efc_frictionloss;     // frictionloss (friction)                  (njmax x 1)
    mjtNum*   efc_pos;              // constraint position (equality, contact)  (njmax x 1)
    mjtNum*   efc_J;                // constraint Jacobian                      (njmax x nv)
    mjtNum*   efc_J_T;              // sparse constraint Jacobian transposed    (nv x njmax)
    mjtNum*   efc_diagApprox;       // approximation to diagonal of A           (njmax x 1)
    mjtNum*   efc_D;                // constraint mass							(njmax x 1)
    mjtNum*   efc_R;                // inverse constraint mass                  (njmax x 1)

    // computed by mj_fwdPosition/mj_projectConstraint; DENSE only
    mjtNum*   efc_AR;               // J*inv(M)*J' + R                          (njmax x njmax)
    mjtNum*   e_ARchol;             // chol(Ae)                                 (nemax x nemax)
    mjtNum*   fc_e_rect;            // Aie*inv(Ae)                              (njmax x nemax)
    mjtNum*   fc_AR;                // Ai - Aie*inv(Ae)*Aei                     (njmax x njmax)

    //-------------------------------- POSITION, VELOCITY dependent

    // computed by mj_fwdVelocity
    mjtNum*   ten_velocity;         // tendon velocities                        (ntendon x 1)
    mjtNum*   actuator_velocity;    // actuator velocities                      (nu x 1)

    // computed by mj_fwdVelocity/mj_comVel
    mjtNum*   cvel;                 // com-based velocity [3D rot; 3D tran]     (nbody x 6)
    mjtNum*   cdof_dot;             // time-derivative of cdof                  (nv x 6)

    // computed by mj_fwdVelocity/mj_rne (without acceleration)
    mjtNum*   qfrc_bias;            // C(qpos,qvel)                             (nv x 1)

    // computed by mj_fwdVelocity/mj_passive
    mjtNum*   qfrc_passive;         // passive force                            (nv x 1)

    // computed by mj_fwdVelocity/mj_referenceConstraint
    mjtNum*   efc_vel;              // velocity in constraint space: J*qvel     (njmax x 1)
    mjtNum*   efc_aref;             // reference pseudo-acceleration            (njmax x 1)

    //-------------------------------- POSITION, VELOCITY, CONTROL/ACCELERATION dependent

    // computed by mj_fwdActuation
    mjtNum*   actuator_force;       // actuator force in actuation space        (nu x 1)
    mjtNum*   qfrc_actuator;        // actuator force                           (nv x 1)

    // computed by mj_fwdAcceleration
    mjtNum*   qfrc_unc;             // net unconstrained force                  (nv x 1)
    mjtNum*   qacc_unc;             // unconstrained acceleration               (nv x 1)

    // computed by mj_fwdConstraint
    mjtNum*   efc_b;                // linear cost term: J*qacc_unc - aref      (njmax x 1)
    mjtNum*   fc_b;                 // bi - Aie*inv(Ae)*be; DENSE only          (njmax x 1)
    mjtNum*   efc_force;            // constraint force in constraint space     (njmax x 1)
    mjtNum*   qfrc_constraint;      // constraint force                         (nv x 1)

    // computed by mj_inverse
    mjtNum*   qfrc_inverse;         // net external force; should equal:        (nv x 1)
                                    //  qfrc_applied + J'*xfrc_applied + qfrc_actuator 

    // computed by mj_sensor/mj_rnePostConstraint; rotation:translation format
    mjtNum*   cacc;                 // com-based acceleration                   (nbody x 6)
    mjtNum*   cfrc_int;             // com-based interaction force with parent  (nbody x 6)
    mjtNum*   cfrc_ext;             // com-based external force on body         (nbody x 6)
};
typedef struct _mjData mjData;


//---------------------------------- callback function types ----------------------------

// generic MuJoCo function
typedef void (*mjfGeneric)(const mjModel* m, mjData* d);

// timer
typedef long long int (*mjfTime)(void);     

// actuator dynamics, gain, bias
typedef mjtNum (*mjfAct)(const mjModel* m, const mjData* d, int id);

// magentic flux at global position
typedef void (*mjfMagnetic)(const mjModel* m, const mjData* d, 
                            const mjtNum* pos, mjtNum* flux);

// solver impedance
typedef mjtNum (*mjfSolImp)(const mjModel* m, const mjData* d, int id, 
                            mjtNum distance, mjtNum* constimp);

// solver reference
typedef void (*mjfSolRef)(const mjModel* m, const mjData* d, int id,
                          mjtNum constimp, mjtNum imp, int dim, mjtNum* ref);

// collision detection
typedef int (*mjfCollision)(const mjModel* m, const mjData* d, 
                            mjContact* con, int g1, int g2, mjtNum margin);
