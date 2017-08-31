//-----------------------------------//
//  This file is part of MuJoCo.     //
//  Copyright (C) 2016 Roboti LLC.   //
//-----------------------------------//


#pragma once


//---------------------------- global constants -----------------------------------------

#define mjNGROUP        5           // number of geom and site groups with visflags
#define mjMAXOVERLAY    500         // maximum number of characters in overlay text


//------------------------------- 3D visualization --------------------------------------

typedef enum _mjtCatBit             // bitflags for mjvGeom category
{
    mjCAT_STATIC        = 1,        // model elements in body 0
    mjCAT_DYNAMIC       = 2,        // model elements in all other bodies
    mjCAT_DECOR         = 4,        // decorative geoms
    mjCAT_ALL           = 7         // select all categories
} mjtCatBit;


typedef enum _mjtMouse              // mouse interaction mode
{
    mjMOUSE_NONE         = 0,       // no action
    mjMOUSE_ROTATE_V,               // rotate, vertical plane
    mjMOUSE_ROTATE_H,               // rotate, horizontal plane
    mjMOUSE_MOVE_V,                 // move, vertical plane
    mjMOUSE_MOVE_H,                 // move, horizontal plane
    mjMOUSE_ZOOM,                   // zoom
    mjMOUSE_SELECT                  // selection
} mjtMouse;


typedef enum _mjtPertBit            // mouse perturbations
{
    mjPERT_TRANSLATE    = 1,        // translation
    mjPERT_ROTATE       = 2         // rotation
} mjtPertBit;


typedef enum _mjtLabel              // object labeling
{
    mjLABEL_NONE        = 0,        // nothing
    mjLABEL_BODY,                   // body labels
    mjLABEL_JOINT,                  // joint labels
    mjLABEL_GEOM,                   // geom labels
    mjLABEL_SITE,                   // site labels
    mjLABEL_CAMERA,                 // camera labels
    mjLABEL_LIGHT,                  // light labels
    mjLABEL_TENDON,                 // tendon labels
    mjLABEL_ACTUATOR,               // actuator labels
    mjLABEL_CONSTRAINT,             // constraint labels
    mjLABEL_SELECTION,              // selected object
    mjLABEL_SELPNT,                 // coordinates of selection point

    mjNLABEL						// number of label types
} mjtLabel;


typedef enum _mjtFrame              // frame visualization
{
    mjFRAME_NONE        = 0,        // no frames
    mjFRAME_BODY,                   // body frames
    mjFRAME_GEOM,                   // geom frames
    mjFRAME_SITE,                   // site frames
    mjFRAME_CAMERA,					// camera frame
    mjFRAME_LIGHT,					// light frame
    mjFRAME_WORLD,                  // world frame

    mjNFRAME						// number of visualization frames
} mjtFrame;


typedef enum _mjtVisFlag            // flags enabling model element visualization
{
    mjVIS_CONVEXHULL    = 0,        // mesh convex hull
    mjVIS_TEXTURE,                  // textures
    mjVIS_JOINT,                    // joints
    mjVIS_ACTUATOR,                 // actuators
    mjVIS_CAMERA,					// cameras
    mjVIS_LIGHT,					// lights
    mjVIS_CONSTRAINT,               // point constraints
    mjVIS_INERTIA,                  // equivalent inertia boxes
    mjVIS_PERTFORCE,                // perturbation force
    mjVIS_PERTOBJ,                  // perturbation object
    mjVIS_CONTACTPOINT,             // contact points
    mjVIS_CONTACTFORCE,             // contact force
    mjVIS_CONTACTSPLIT,             // split contact force into normal and tanget
    mjVIS_TRANSPARENT,              // make dynamic geoms more transparent
    mjVIS_AUTOCONNECT,              // auto connect joints and body coms
    mjVIS_COM,                      // center of mass
    mjVIS_SELECT,                   // selection point
    mjVIS_STATIC,                   // static bodies

    mjNVISFLAG                      // number of visualization flags
} mjtVisFlag;


struct _mjvGeom                     // all info needed to specify one abstract geom
{
    // type info
    int     type;                   // geom type (mjtGeom)
    int     dataid;                 // mesh, hfield or plane id; -1: none
    int     objtype;                // mujoco object type; mjOBJ_UNKNOWN for decor
    int     objid;                  // mujoco object id; -1 for decor
    int     category;               // visual category
    int     texid;                  // texture id; -1: no texture
    int     texuniform;             // uniform cube mapping

    // OpenGL info
    float   texrepeat[2];           // texture repetition for 2D mapping
    float   size[3];                // size parameters
    float   pos[3];                 // Cartesian position
    float   mat[9];                 // Cartesian orientation
    float   rgba[4];                // color and transparency
    float   emission;               // emission coef
    float   specular;               // specular coef
    float   shininess;              // shininess coef
    float   reflectance;            // reflectance coef
    char    label[100];             // text label

    // transparency rendering (set internally)
    float   camdist;                // distance to camera (used by sorter)
    float   rbound;                 // rbound if known, otherwise 0
    mjtByte transparent;            // treat geom as transparent
};
typedef struct _mjvGeom mjvGeom;


struct _mjvOption                   // visualization options, window-specific
{
    int     label;                  // what objects to label (mjtLabel)
    int     frame;                  // which frame to show (mjtFrame)
    mjtByte geomgroup[mjNGROUP];    // geom visualization by group
    mjtByte sitegroup[mjNGROUP];    // site visualization by group
    mjtByte flags[mjNVISFLAG];      // visualization flags (indexed by mjtVisFlag)
};
typedef struct _mjvOption mjvOption;


struct _mjvCameraPose               // low-level camera parameters
{
    mjtNum  head_pos[3];            // head position
    mjtNum  head_right[3];          // head right axis
    mjtNum  window_pos[3];          // window center position
    mjtNum  window_right[3];        // window/monitor right axis
    mjtNum  window_up[3];           // window/monitor up axis
    mjtNum  window_normal[3];       // window/monitor normal axis
    mjtNum  window_size[2];         // physical window size
    mjtNum  scale;                  // uniform model scaling rel.to origin
    mjtNum  ipd;                    // inter-pupilary distance
};
typedef struct _mjvCameraPose mjvCameraPose;


struct _mjvCamera                   // camera control, window-specific
{
    // constant parameters
    mjtNum  fovy;                   // y-field of view (deg)

    // camera id, trackbody id
    int     camid;                  // fixed camera id; -1: free
    int     trackbodyid;            // body id to track; -1: no tracking

    // free camera parameters, used to compute camera pose
    mjtNum  lookat[3];              // where the camera is looking
    mjtNum  azimuth;                // camera azimuth (in DEG)
    mjtNum  elevation;              // camera elevation (in DEG)
    mjtNum  distance;               // distance to lookat point

    // physical parameters that determine actual rendering
    mjvCameraPose pose;             // head, window, scale, ipd

    mjtByte VR;                     // VR mode: use pose directly
};
typedef struct _mjvCamera mjvCamera;


struct _mjvLight                    // light
{
    float   pos[3];                 // position rel. to body frame              
    float   dir[3];                 // direction rel. to body frame             
    float   attenuation[3];         // OpenGL attenuation (quadratic model)     
    float   cutoff;                 // OpenGL cutoff                            
    float   exponent;               // OpenGL exponent                          
    float   ambient[3];             // ambient rgb (alpha=1)                    
    float   diffuse[3];             // diffuse rgb (alpha=1)                    
    float   specular[3];            // specular rgb (alpha=1)
    mjtByte headlight;              // headlight
    mjtByte directional;            // directional light                        
    mjtByte castshadow;             // does light cast shadows                  
};
typedef struct _mjvLight mjvLight;


struct _mjvObjects                  // collection of abstract visualization objects
{
    int nlight;                     // number of lights currently in buffer
    int ngeom;                      // number of geoms currently in buffer
    int maxgeom;                    // size of allocated geom buffer

    mjvLight lights[8];             // buffer for lights
    mjvGeom* geoms;                 // buffer for geoms; managed by mjv_make/clearObjects
    int*     geomorder;             // buffer for ordering geoms by distance to camera
};
typedef struct _mjvObjects mjvObjects;
