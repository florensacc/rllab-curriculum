include "mjmodel.pxd"
include "mjdata.pxd"
include "mjrender.pxd"
include "mjvisualize.pxd"


cdef extern from "mujoco.h" nogil:
    # macros
    #define mjMARKSTACK   int _mark = d->pstack;
    #define mjFREESTACK   d->pstack = _mark;
    #define mjDISABLED(x) (m->opt.disableflags & (x))
    #define mjENABLED(x)  (m->opt.enableflags & (x))


    # user error and memory handlers
    void  (*mju_user_error)(const char*);
    void  (*mju_user_warning)(const char*);
    void* (*mju_user_malloc)(size_t);
    void  (*mju_user_free)(void*);


    # # callbacks extending computation pipeline
    # mjfGeneric  mjcb_passive;
    # mjfGeneric  mjcb_control;
    # mjfSensor   mjcb_sensor;
    # mjfTime     mjcb_time;
    # mjfAct      mjcb_act_dyn;
    # mjfAct      mjcb_act_gain;
    # mjfAct      mjcb_act_bias;
    # mjfSolImp   mjcb_sol_imp;
    # mjfSolRef   mjcb_sol_ref;
    #
    #
    # # collision function table
    # mjfCollision mjCOLLISIONFUNC[mjNGEOMTYPES][mjNGEOMTYPES];
    #
    #
    # # string names
    const char* mjDISABLESTRING[mjNDISABLE];
    const char* mjENABLESTRING[mjNENABLE];
    const char* mjTIMERSTRING[mjNTIMER];
    const char* mjLABELSTRING[mjNLABEL];
    const char* mjFRAMESTRING[mjNFRAME];
    const char* mjVISSTRING[mjNVISFLAG][3];
    const char* mjRNDSTRING[mjNRNDFLAG][3];


    #---------------------- License activation and certificate (mutex-protected) -----------

    # activate license, call mju_error on failure; return 1 if ok, 0 if failure
    int mj_activate(const char* filename);

    # deactivate license, free memory
    void mj_deactivate();

    # server: generate certificate question
    void mj_certQuestion(mjtNum question[16]);

    # client: generate certificate answer given question
    void mj_certAnswer(const mjtNum question[16], mjtNum answer[16]);

    # server: check certificate question-answer pair; return 1 if match, 0 if mismatch
    int mj_certCheck(const mjtNum question[16], const mjtNum answer[16]);


    #---------------------- XML parser and C++ compiler (mutex-protected) ------------------

    # parse XML file or string in MJCF or URDF format, compile it, return low-level model
    #  if xmlstring is not NULL, it has precedence over filename
    #  error can be NULL; otherwise assumed to have size error_sz
    mjModel* mj_loadXML(const char* filename, const char* xmlstring,
                              char* error, int error_sz);

    # update XML data structures with info from low-level model, save as MJCF
    #  error can be NULL; otherwise assumed to have size error_sz
    int mj_saveXML(const char* filename, const mjModel* m, char* error, int error_sz);

    # print internal XML schema as plain text or HTML, with style-padding or &nbsp;
    int mj_printSchema(const char* filename, char* buffer, int buffer_sz,
                             int flg_html, int flg_pad);


    #---------------------- Main entry points ----------------------------------------------

    # advance simulation: use control callback, no external force, RK4 available
    void mj_step(const mjModel* m, mjData* d);

    # advance simulation in two steps: before external force/control is set by user
    void mj_step1(const mjModel* m, mjData* d);

    # advance simulation in two steps: after external force/control is set by user
    void mj_step2(const mjModel* m, mjData* d);

    # forward dynamics
    void mj_forward(const mjModel* m, mjData* d);

    # inverse dynamics
    void mj_inverse(const mjModel* m, mjData* d);

    # forward dynamics with skip; skipstage is mjtStage
    void mj_forwardSkip(const mjModel* m, mjData* d,
                              int skipstage, int skipsensorenergy);

    # inverse dynamics with skip; skipstage is mjtStage
    void mj_inverseSkip(const mjModel* m, mjData* d,
                              int skipstage, int skipsensorenergy);


    #---------------------- Model and data initialization ----------------------------------

    # set default solver parameters
    void mj_defaultSolRefImp(mjtNum* solref, mjtNum* solimp);

    # # set physics options to default values
    # void mj_defaultOption(mjOption* opt);
    #
    # # set visual options to default values
    # void mj_defaultVisual(mjVisual* vis);

    # copy mjModel; allocate new if dest is NULL
    mjModel* mj_copyModel(mjModel* dest, const mjModel* src);

    # save model to binary file or memory buffer (buffer has precedence if szbuf>0)
    void mj_saveModel(const mjModel* m, const char* filename, void* buffer, int buffer_sz);

    # load model from binary file or memory buffer (buffer has precedence if szbuf>0)
    mjModel* mj_loadModel(const char* filename, void* buffer, int buffer_sz);

    # de-allocate model
    void mj_deleteModel(mjModel* m);

    # size of buffer needed to hold model
    int mj_sizeModel(const mjModel* m);

    # allocate mjData correponding to given model
    mjData* mj_makeData(const mjModel* m);

    # copy mjData
    mjData* mj_copyData(mjData* dest, const mjModel* m, const mjData* src);

    # set data to defaults
    void mj_resetData(const mjModel* m, mjData* d);

    # set data to defaults, fill everything else with debug_value
    void mj_resetDataDebug(const mjModel* m, mjData* d, unsigned char debug_value);

    # reset data, set fields from specified keyframe
    void mj_resetDataKeyframe(const mjModel* m, mjData* d, int key);

    # mjData stack allocate
    mjtNum* mj_stackAlloc(mjData* d, int size);

    # de-allocate data
    void mj_deleteData(mjData* d);

    # reset callbacks to defaults
    void mj_resetCallbacks();

    # set constant fields of mjModel
    void mj_setConst(mjModel* m, mjData* d, int flg_actrange);


    #---------------------- Printing -------------------------------------------------------

    # print model to text file
    void mj_printModel(const mjModel* m, const char* filename);

    # print data to text file
    void mj_printData(const mjModel* m, mjData* d, const char* filename);

    # print matrix to screen
    void mju_printMat(const mjtNum* mat, int nr, int nc);


    #---------------------- Components: forward dynamics -----------------------------------

    # position-dependent computations
    void mj_fwdPosition(const mjModel* m, mjData* d);

    # velocity-dependent computations
    void mj_fwdVelocity(const mjModel* m, mjData* d);

    # compute actuator force
    void mj_fwdActuation(const mjModel* m, mjData* d);

    # add up all non-constraint forces, compute qacc_unc
    void mj_fwdAcceleration(const mjModel* m, mjData* d);

    # constraint solver
    void mj_fwdConstraint(const mjModel* m, mjData* d);

    # Euler integrator, semi-implicit in velocity
    void mj_Euler(const mjModel* m, mjData* d);

    # Runge-Kutta explicit order-N integrator
    void mj_RungeKutta(const mjModel* m, mjData* d, int N);


    #---------------------- Components: inverse dynamics -----------------------------------

    # position-dependent computations
    void mj_invPosition(const mjModel* m, mjData* d);

    # velocity-dependent computations
    void mj_invVelocity(const mjModel* m, mjData* d);

    # constraint solver
    void mj_invConstraint(const mjModel* m, mjData* d);

    # compare forward and inverse dynamics, without changing results of forward dynamics
    void mj_compareFwdInv(const mjModel* m, mjData* d);


    #---------------------- Components: forward and inverse dynamics -----------------------

    # position-dependent sensors
    void mj_sensorPos(const mjModel* m, mjData* d);

    # velocity-dependent sensors
    void mj_sensorVel(const mjModel* m, mjData* d);

    # acceleration/force-dependent sensors
    void mj_sensorAcc(const mjModel* m, mjData* d);

    # position-dependent energy (potential)
    void mj_energyPos(const mjModel* m, mjData* d);

    # velocity-dependent energy (kinetic)
    void mj_energyVel(const mjModel* m, mjData* d);


    #---------------------- Sub-components -------------------------------------------------

    # check positions; reset if bad
    void mj_checkPos(const mjModel* m, mjData* d);

    # check velocities; reset if bad
    void mj_checkVel(const mjModel* m, mjData* d);

    # check accelerations; reset if bad
    void mj_checkAcc(const mjModel* m, mjData* d);

    # forward kinematics
    void mj_kinematics(const mjModel* m, mjData* d);

    # map inertias and motion dofs to global frame centered at CoM
    void mj_comPos(const mjModel* m, mjData* d);

    # compute camera and light positions and orientations
    void mj_camlight(const mjModel* m, mjData* d);

    # compute tendon lengths, velocities and moment arms
    void mj_tendon(const mjModel* m, mjData* d);

    # compute actuator transmission lengths and moments
    void mj_transmission(const mjModel* m, mjData* d);

    # composite rigid body inertia algorithm
    void mj_crb(const mjModel* m, mjData* d);

    # sparse L'*D*L factorizaton of the inertia matrix
    void mj_factorM(const mjModel* m, mjData* d);

    # sparse backsubstitution:  x = inv(L'*D*L)*y
    void mj_backsubM(const mjModel* m, mjData* d, mjtNum* x, const mjtNum* y, int n);

    # half of sparse backsubstitution:  x = sqrt(inv(D))*inv(L')*y
    void mj_backsubM2(const mjModel* m, mjData* d, mjtNum* x, const mjtNum* y, int n);

    # compute cvel, cdof_dot
    void mj_comVel(const mjModel* m, mjData* d);

    # spring-dampers and body viscosity
    void mj_passive(const mjModel* m, mjData* d);

    # RNE: compute M(qpos)*qacc + C(qpos,qvel); flg_acc=0 removes inertial term
    void mj_rne(const mjModel* m, mjData* d, int flg_acc, mjtNum* result);

    # RNE with complete data: compute cacc, cfrc_ext, cfrc_int
    void mj_rnePostConstraint(const mjModel* m, mjData* d);

    # collision detection
    void mj_collision(const mjModel* m, mjData* d);

    # construct constraints
    void mj_makeConstraint(const mjModel* m, mjData* d);

    # compute dense matrices: efc_AR, e_ARchol, fc_half, fc_AR
    void mj_projectConstraint(const mjModel* m, mjData* d);

    # compute efc_vel, efc_aref
    void mj_referenceConstraint(const mjModel* m, mjData* d);


    #---------------------- Support functions ----------------------------------------------

    # add contact to d->contact list; return 0 if success; 1 if buffer full
    int mj_addContact(const mjModel* m, mjData* d, const mjContact* con);

    # determine type of friction cone
    int mj_isPyramid(const mjModel* m);

    # determine type of constraint Jacobian
    int mj_isSparse(const mjModel* m);

    # multiply Jacobian by vector
    void mj_mulJacVec(const mjModel* m, mjData* d,
                            mjtNum* res, const mjtNum* vec);

    # multiply JacobianT by vector
    void mj_mulJacTVec(const mjModel* m, mjData* d, mjtNum* res, const mjtNum* vec);

    # compute 3/6-by-nv Jacobian of global point attached to given body
    void mj_jac(const mjModel* m, const mjData* d,
                      mjtNum* jacp, mjtNum* jacr, const mjtNum* point, int body);

    # compute body frame Jacobian
    void mj_jacBody(const mjModel* m, const mjData* d,
                          mjtNum* jacp, mjtNum* jacr, int body);

    # compute body center-of-mass Jacobian
    void mj_jacBodyCom(const mjModel* m, const mjData* d,
                             mjtNum* jacp, mjtNum* jacr, int body);

    # compute geom Jacobian
    void mj_jacGeom(const mjModel* m, const mjData* d,
                          mjtNum* jacp, mjtNum* jacr, int geom);

    # compute site Jacobian
    void mj_jacSite(const mjModel* m, const mjData* d,
                          mjtNum* jacp, mjtNum* jacr, int site);

    # compute translation Jacobian of point, and rotation Jacobian of axis
    void mj_jacPointAxis(const mjModel* m, mjData* d,
                               mjtNum* jacPoint, mjtNum* jacAxis,
                               const mjtNum* point, const mjtNum* axis, int body);

    # get id of object with specified name; -1: not found; type is mjtObj
    int mj_name2id(const mjModel* m, int type, const char* name);

    # get name of object with specified id; 0: invalid type or id; type is mjtObj
    const char* mj_id2name(const mjModel* m, int type, int id);

    # convert sparse inertia matrix M into full matrix
    void mj_fullM(const mjModel* m, mjtNum* dst, const mjtNum* M);

    # multiply vector by inertia matrix
    void mj_mulM(const mjModel* m, const mjData* d, mjtNum* res, const mjtNum* vec);

    # apply cartesian force and torque (outside xfrc_applied mechanism)
    void mj_applyFT(const mjModel* m, mjData* d,
                          const mjtNum* force, const mjtNum* torque,
                          const mjtNum* point, int body, mjtNum* qfrc_target);

    # compute object 6D velocity in object-centered frame, world/local orientation
    void mj_objectVelocity(const mjModel* m, const mjData* d,
                                 int objtype, int objid, mjtNum* res, int flg_local);

    # compute object 6D acceleration in object-centered frame, world/local orientation
    void mj_objectAcceleration(const mjModel* m, const mjData* d,
                                     int objtype, int objid, mjtNum* res, int flg_local);

    # compute velocity by finite-differencing two positions
    void mj_differentiatePos(const mjModel* m, mjtNum* qvel, mjtNum dt,
                                   const mjtNum* qpos1, const mjtNum* qpos2);

    # extract 6D force:torque for one contact, in contact frame
    void mj_contactForce(const mjModel* m, const mjData* d, int id, mjtNum* result);

    # integrate position with given velocity
    void mj_integratePos(const mjModel* m, mjtNum* qpos, const mjtNum* qvel, mjtNum dt);

    # normalize all quaterions in qpos-type vector
    void mj_normalizeQuat(const mjModel* m, mjtNum* qpos);

    # map from body local to global Cartesian coordinates
    void mj_local2Global(mjData* d, mjtNum* xpos, mjtNum* xmat,
                               const mjtNum* pos, const mjtNum* quat, int body);

    # sum all body masses
    mjtNum mj_getTotalmass(const mjModel* m);

    # scale body masses and inertias to achieve specified total mass
    void mj_setTotalmass(mjModel* m, mjtNum newmass);

    # version number: 1.0.2 is encoded as 102
    int mj_version();


    #---------------------- Abstract interaction -------------------------------------------

    # set default camera
    void mjv_defaultCamera(mjvCamera* cam);

    # set default perturbation
    void mjv_defaultPerturb(mjvPerturb* pert);

    # transform pose from room to model space
    void mjv_room2model(mjtNum* modelpos, mjtNum* modelquat, const mjtNum* roompos,
                              const mjtNum* roomquat, const mjvScene* scn);

    # transform pose from model to room space
    void mjv_model2room(mjtNum* roompos, mjtNum* roomquat, const mjtNum* modelpos,
                              const mjtNum* modelquat, const mjvScene* scn);

    # get camera info in model space: average left and right OpenGL cameras
    void mjv_cameraInModel(mjtNum* headpos, mjtNum* forward, const mjvScene* scn);

    # get camera info in room space: average left and right OpenGL cameras
    void mjv_cameraInRoom(mjtNum* headpos, mjtNum* forward, const mjvScene* scn);

    # get frustum height at unit distance from camera; average left and right OpenGL cameras
    mjtNum mjv_frustumHeight(const mjvScene* scn);

    # rotate 3D vec in horizontal plane by angle between (0,1) and (forward_x,forward_y)
    void mjv_alignToCamera(mjtNum* res, const mjtNum* vec, const mjtNum* forward);

    # move camera with mouse; action is mjtMouse
    void mjv_moveCamera(const mjModel* m, int action, mjtNum reldx, mjtNum reldy,
                              const mjvScene* scn, mjvCamera* cam);

    # move perturb object with mouse; action is mjtMouse
    void mjv_movePerturb(const mjModel* m, const mjData* d, int action, mjtNum reldx,
                               mjtNum reldy, const mjvScene* scn, mjvPerturb* pert);

    # move model with mouse; action is mjtMouse
    void mjv_moveModel(const mjModel* m, int action, mjtNum reldx, mjtNum reldy,
                             const mjtNum* roomup, mjvScene* scn);

    # copy perturb pos,quat from selected body; set scale for perturbation
    void mjv_initPerturb(const mjModel* m, const mjData* d,
                               const mjvScene* scn, mjvPerturb* pert);

    # set perturb pos,quat in d->mocap when selected body is mocap, and in d->qpos otherwise
    #  d->qpos written only if flg_paused and subtree root for selected body has free joint
    void mjv_applyPerturbPose(const mjModel* m, mjData* d, const mjvPerturb* pert,
                                    int flg_paused);

    # set perturb force,torque in d->xfrc_applied, if selected body is dynamic
    void mjv_applyPerturbForce(const mjModel* m, mjData* d, const mjvPerturb* pert);


    #---------------------- Asbtract visualization -----------------------------------------

    # set default visualization options
    void mjv_defaultOption(mjvOption* opt);

    # allocate and init abstract scene
    void mjv_makeScene(mjvScene* scn, int maxgeom);

    # free abstract scene
    void mjv_freeScene(mjvScene* scn);

    # update entire scene
    void mjv_updateScene(const mjModel* m, mjData* d, const mjvOption* opt,
                               const mjvPerturb* pert, mjvCamera* cam, int catmask, mjvScene* scn);

    # add geoms from selected categories to existing scene
    void mjv_addGeoms(const mjModel* m, mjData* d, const mjvOption* opt,
                            const mjvPerturb* pert, int catmask, mjvScene* scn);

    # update camera only
    void mjv_updateCamera(const mjModel* m, mjData* d, mjvCamera* cam, mjvScene* scn);


    #---------------------- OpenGL rendering -----------------------------------------------

    # set default mjrContext
    void mjr_defaultContext(mjrContext* con);

    # allocate resources in custom OpenGL context; fontscale is mjtFontScale
    void mjr_makeContext(const mjModel* m, mjrContext* con, int fontscale);

    # free resources in custom OpenGL context, set to default
    void mjr_freeContext(mjrContext* con);

    # (re) upload texture to GPU
    void mjr_uploadTexture(const mjModel* m, const mjrContext* con, int texid);

    # (re) upload mesh to GPU
    void mjr_uploadMesh(const mjModel* m, const mjrContext* con, int meshid);

    # (re) upload height field to GPU
    void mjr_uploadHField(const mjModel* m, const mjrContext* con, int hfieldid);

    # set OpenGL framebuffer for rendering: mjFB_WINDOW or mjFB_OFFSCREEN
    #  if only one buffer is available, set that buffer and ignore framebuffer argument
    void mjr_setBuffer(int framebuffer, mjrContext* con);

    # read pixels from current OpenGL framebuffer to client buffer
    #  viewport is in OpenGL framebuffer; client buffer starts at (0,0)
    void mjr_readPixels(unsigned char* rgb, float* depth,
                              mjrRect viewport, const mjrContext* con);

    # draw pixels from client buffer to current OpenGL framebuffer
    #  viewport is in OpenGL framebuffer; client buffer starts at (0,0)
    void mjr_drawPixels(const unsigned char* rgb, const float* depth,
                              mjrRect viewport, const mjrContext* con);

    # blit from src viewpoint in current framebuffer to dst viewport in other framebuffer
    #  if src, dst have different size and flg_depth==0, color is interpolated with GL_LINEAR
    void mjr_blitBuffer(mjrRect src, mjrRect dst,
                              int flg_color, int flg_depth, const mjrContext* con);

    # draw text at (x,y) in relative coordinates; font is mjtFont
    void mjr_text(int font, const char* txt, const mjrContext* con,
                        float x, float y, float r, float g, float b);

    # draw text overlay; font is mjtFont; gridpos is mjtGridPos
    void mjr_overlay(int font, int gridpos, mjrRect viewport,
                           const char* overlay, const char* overlay2, const mjrContext* con);

    # get maximum viewport for active buffer
    mjrRect mjr_maxViewport(const mjrContext* con);

    # draw rectangle
    void mjr_rectangle(mjrRect viewport, float r, float g, float b, float a);

    # draw lines
    void mjr_lines(mjrRect viewport, int nline, const float* rgb,
                         const int* npoint, const mjtNum* data);

    # 3D rendering
    void mjr_render(mjrRect viewport, mjvScene* scn, const mjrContext* con);

    # 3D selection
    int mjr_select(mjrRect viewport, const mjvScene* scn, const mjrContext* con,
                         int mousex, int mousey, mjtNum* pos, mjtNum* depth);

    # call glFinish
    void mjr_finish();

    # call glGetError and return result
    int mjr_getError();


    #---------------------- Utility functions: error and memory ----------------------------

    # main error function; does not return to caller
    void mju_error(const char* msg);

    # error function with int argument; msg is a printf format string
    void mju_error_i(const char* msg, int i);

    # error function with string argument
    void mju_error_s(const char* msg, const char* text);

    # main warning function; returns to caller
    void mju_warning(const char* msg);

    # warning function with int argument
    void mju_warning_i(const char* msg, int i);

    # warning function with string argument
    void mju_warning_s(const char* msg, const char* text);

    # clear user error and memory handlers
    void mju_clearHandlers();

    # allocate memory; byte-align on 8; pad size to multiple of 8
    void* mju_malloc(size_t size);

    # free memory (with free() by default)
    void mju_free(void* ptr);

    # high-level warning function: count warnings in mjData, print only the first
    void mj_warning(mjData* d, int warning, int info);


    #---------------------- Utility functions: basic math ----------------------------------

    #define mjMAX(a,b) (((a) > (b)) ? (a) : (b))
    #define mjMIN(a,b) (((a) < (b)) ? (a) : (b))

    #ifdef mjUSEDOUBLE
        #define mju_sqrt    sqrt
        #define mju_exp     exp
        #define mju_sin     sin
        #define mju_cos     cos
        #define mju_tan     tan
        #define mju_asin    asin
        #define mju_acos    acos
        #define mju_atan2   atan2
        #define mju_tanh    tanh
        #define mju_pow     pow
        #define mju_abs     fabs
        #define mju_log     log
        #define mju_log10   log10
        #define mju_floor   floor
        #define mju_ceil    ceil

    #else
        #define mju_sqrt    sqrtf
        #define mju_exp     expf
        #define mju_sin     sinf
        #define mju_cos     cosf
        #define mju_tan     tanf
        #define mju_asin    asinf
        #define mju_acos    acosf
        #define mju_atan2   atan2f
        #define mju_tanh    tanhf
        #define mju_pow     powf
        #define mju_abs     fabsf
        #define mju_log     logf
        #define mju_log10   log10f
        #define mju_floor   floorf
        #define mju_ceil    ceilf
    #endif

    # set vector to zero
    void mju_zero3(mjtNum* res);

    # copy vector
    void mju_copy3(mjtNum* res, const mjtNum* data);

    # scale vector
    void mju_scl3(mjtNum* res, const mjtNum* vec, mjtNum scl);

    # add vectors
    void mju_add3(mjtNum* res, const mjtNum* vec1, const mjtNum* vec2);

    # subtract vectors
    void mju_sub3(mjtNum* res, const mjtNum* vec1, const mjtNum* vec2);

    # add to vector
    void mju_addTo3(mjtNum* res, const mjtNum* vec);

    # add scaled to vector
    void mju_addToScl3(mjtNum* res, const mjtNum* vec, mjtNum scl);

    # res = vec1 + scl*vec2
    void mju_addScl3(mjtNum* res, const mjtNum* vec1, const mjtNum* vec2, mjtNum scl);

    # normalize vector, return length before normalization
    mjtNum mju_normalize3(mjtNum* res);

    # compute vector length (without normalizing)
    mjtNum mju_norm3(const mjtNum* res);

    # vector dot-product
    mjtNum mju_dot3(const mjtNum* vec1, const mjtNum* vec2);

    # Cartesian distance between 3D vectors
    mjtNum mju_dist3(const mjtNum* pos1, const mjtNum* pos2);

    # multiply vector by 3D rotation matrix
    void mju_rotVecMat(mjtNum* res, const mjtNum* vec, const mjtNum* mat);

    # multiply vector by transposed 3D rotation matrix
    void mju_rotVecMatT(mjtNum* res, const mjtNum* vec, const mjtNum* mat);

    # vector cross-product, 3D
    void mju_cross(mjtNum* res, const mjtNum* a, const mjtNum* b);

    # set vector to zero
    void mju_zero(mjtNum* res, int n);

    # copy vector
    void mju_copy(mjtNum* res, const mjtNum* data, int n);

    # scale vector
    void mju_scl(mjtNum* res, const mjtNum* vec, mjtNum scl, int n);

    # add vectors
    void mju_add(mjtNum* res, const mjtNum* vec1, const mjtNum* vec2, int n);

    # subtract vectors
    void mju_sub(mjtNum* res, const mjtNum* vec1, const mjtNum* vec2, int n);

    # add to vector
    void mju_addTo(mjtNum* res, const mjtNum* vec, int n);

    # add scaled to vector
    void mju_addToScl(mjtNum* res, const mjtNum* vec, mjtNum scl, int n);

    # res = vec1 + scl*vec2
    void mju_addScl(mjtNum* res, const mjtNum* vec1, const mjtNum* vec2,
                          mjtNum scl, int n);

    # normalize vector, return length before normalization
    mjtNum mju_normalize(mjtNum* res, int n);

    # compute vector length (without normalizing)
    mjtNum mju_norm(const mjtNum* res, int n);

    # vector dot-product
    mjtNum mju_dot(const mjtNum* vec1, const mjtNum* vec2, const int n);

    # multiply matrix and vector
    void mju_mulMatVec(mjtNum* res, const mjtNum* mat, const mjtNum* vec,
                             int nr, int nc);

    # multiply transposed matrix and vector
    void mju_mulMatTVec(mjtNum* res, const mjtNum* mat, const mjtNum* vec,
                              int nr, int nc);

    # transpose matrix
    void mju_transpose(mjtNum* res, const mjtNum* mat, int r, int c);

    # multiply matrices
    void mju_mulMatMat(mjtNum* res, const mjtNum* mat1, const mjtNum* mat2,
                             int r1, int c1, int c2);

    # multiply matrices, second argument transposed
    void mju_mulMatMatT(mjtNum* res, const mjtNum* mat1, const mjtNum* mat2,
                              int r1, int c1, int r2);

    # multiply matrices, first argument transposed
    void mju_mulMatTMat(mjtNum* res, const mjtNum* mat1, const mjtNum* mat2,
                              int r1, int c1, int c2);

    # compute M*M'; scratch must be at least r*c
    void mju_sqrMat(mjtNum* res, const mjtNum* mat, int r, int c,
                          mjtNum* scratch, int nscratch);

    # compute M'*diag*M (diag=NULL: compute M'*M)
    void mju_sqrMatTD(mjtNum* res, const mjtNum* mat, const mjtNum* diag, int r, int c);

    # coordinate transform of 6D motion or force vector in rotation:translation format
    #  rotnew2old is 3-by-3, NULL means no rotation; flg_force specifies force or motion type
    void mju_transformSpatial(mjtNum* res, const mjtNum* vec, int flg_force,
                                    const mjtNum* newpos, const mjtNum* oldpos,
                                    const mjtNum* rotnew2old);


    #---------------------- Utility functions: quaternions ---------------------------------

    # rotate vector by quaternion
    void mju_rotVecQuat(mjtNum* res, const mjtNum* vec, const mjtNum* quat);

    # negate quaternion
    void mju_negQuat(mjtNum* res, const mjtNum* quat);

    # muiltiply quaternions
    void mju_mulQuat(mjtNum* res, const mjtNum* quat1, const mjtNum* quat2);

    # muiltiply quaternion and axis
    void mju_mulQuatAxis(mjtNum* res, const mjtNum* quat, const mjtNum* axis);

    # convert axisAngle to quaternion
    void mju_axisAngle2Quat(mjtNum* res, const mjtNum* axis, mjtNum angle);

    # convert quaternion (corresponding to orientation difference) to 3D velocity
    void mju_quat2Vel(mjtNum* res, const mjtNum* quat, mjtNum dt);

    # convert quaternion to 3D rotation matrix
    void mju_quat2Mat(mjtNum* res, const mjtNum* quat);

    # convert 3D rotation matrix to quaterion
    void mju_mat2Quat(mjtNum* quat, const mjtNum* mat);

    # time-derivative of quaternion, given 3D rotational velocity
    void mju_derivQuat(mjtNum* res, const mjtNum* quat, const mjtNum* vel);

    # integrate quaterion given 3D angular velocity
    void mju_quatIntegrate(mjtNum* quat, const mjtNum* vel, mjtNum scale);

    # compute quaternion performing rotation from z-axis to given vector
    void mju_quatZ2Vec(mjtNum* quat, const mjtNum* vec);


    #---------------------- Utility functions: poses (pos, quat) ---------------------------

    # multiply two poses
    void mju_mulPose(mjtNum* posres, mjtNum* quatres, const mjtNum* pos1,
                           const mjtNum* quat1, const mjtNum* pos2, const mjtNum* quat2);

    # negate pose
    void mju_negPose(mjtNum* posres, mjtNum* quatres,
                           const mjtNum* pos, const mjtNum* quat);

    # transform vector by pose
    void mju_trnVecPose(mjtNum* res, const mjtNum* pos, const mjtNum* quat,
                              const mjtNum* vec);


    #---------------------- Utility functions: matrix decomposition ------------------------

    # Cholesky decomposition
    int mju_cholFactor(mjtNum* mat, mjtNum* diag, int n,
                             mjtNum minabs, mjtNum minrel, mjtNum* correct);

    # Cholesky backsubstitution: phase&i enables forward(i=1), backward(i=2) pass
    void mju_cholBacksub(mjtNum* res, const mjtNum* mat, const mjtNum* vec,
                               int n, int nvec, int phase);

    # eigenvalue decomposition of symmetric 3x3 matrix
    int mju_eig3(mjtNum* eigval, mjtNum* eigvec, mjtNum* quat, const mjtNum* mat);


    #---------------------- Utility functions: miscellaneous -------------------------------

    # muscle FVL curve: prm = (lminrel, lmaxrel, widthrel, vmaxrel, fmax, fvsat)
    mjtNum mju_muscleFVL(mjtNum len, mjtNum vel, mjtNum lmin, mjtNum lmax, mjtNum* prm);

    # muscle passive force: prm = (lminrel, lmaxrel, fpassive)
    mjtNum mju_musclePassive(mjtNum len, mjtNum lmin, mjtNum lmax, mjtNum* prm);

    # pneumatic cylinder dynamics
    mjtNum mju_pneumatic(mjtNum len, mjtNum len0, mjtNum vel, mjtNum* prm,
                               mjtNum act, mjtNum ctrl, mjtNum timestep, mjtNum* jac);

    # convert contact force to pyramid representation
    void mju_encodePyramid(mjtNum* pyramid, const mjtNum* force,
                                 const mjtNum* mu, int dim);

    # convert pyramid representation to contact force
    void mju_decodePyramid(mjtNum* force, const mjtNum* pyramid,
                                 const mjtNum* mu, int dim);

    # integrate spring-damper analytically, return pos(dt)
    mjtNum mju_springDamper(mjtNum pos0, mjtNum vel0, mjtNum Kp, mjtNum Kv, mjtNum dt);

    # min function, single evaluation of a and b
    mjtNum mju_min(mjtNum a, mjtNum b);

    # max function, single evaluation of a and b
    mjtNum mju_max(mjtNum a, mjtNum b);

    # sign function
    mjtNum mju_sign(mjtNum x);

    # round to nearest integer
    int mju_round(mjtNum x);

    # convert type id (mjtObj) to type name
    const char* mju_type2Str(int type);

    # convert type name to type id (mjtObj)
    int mju_str2Type(const char* str);

    # warning text
    const char* mju_warningText(int warning, int info);

    # return 1 if nan or abs(x)>mjMAXVAL, 0 otherwise
    int mju_isBad(mjtNum x);

    # return 1 if all elements are 0
    int mju_isZero(mjtNum* vec, int n);

    # standard normal random number generator (optional second number)
    mjtNum mju_standardNormal(mjtNum* num2);

    # convert from float to mjtNum
    void mju_f2n(mjtNum* res, const float* vec, int n);

    # convert from mjtNum to float
    void mju_n2f(float* res, const mjtNum* vec, int n);

    # convert from double to mjtNum
    void mju_d2n(mjtNum* res, const double* vec, int n);

    # convert from mjtNum to double
    void mju_n2d(double* res, const mjtNum* vec, int n);
