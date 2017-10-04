//-----------------------------------//
//  This file is part of MuJoCo.     //
//  Copyright (C) 2016 Roboti LLC.   //
//-----------------------------------//


#pragma once


typedef enum _mjtGridPos            // grid position for overlay
{
    mjGRID_TOPLEFT      = 0,        // top left
    mjGRID_TOPRIGHT,                // top right
    mjGRID_BOTTOMLEFT,              // bottom left
    mjGRID_BOTTOMRIGHT              // bottom right
} mjtGridPos;


typedef enum _mjtRndFlag            // flags enabling rendering effects
{
    mjRND_SHADOW        = 0,        // shadows
    mjRND_FLIP,                     // flip left/right
    mjRND_REFLECTION,               // reflections
    mjRND_WIREFRAME,                // wireframe
    mjRND_SKYBOX,                   // skybox
    mjRND_FOG,                      // fog

    mjNRNDFLAG                      // number of rendering flags
} mjtRndFlag;


struct _mjrContext                  // custom OpenGL context
{
    float linewidth;                // line width for wireframe rendering
    float znear;                    // near clipping plane
    float zfar;                     // far clipping plane
    float shadowclip;               // clipping radius for directional lights
    float shadowscale;              // fraction of light cutoff for spot lights
    int   shadowsize;               // size of shadow map texture

    unsigned int offwidth;          // width of offscreen buffer
    unsigned int offheight;         // height of offscreen buffer
    unsigned int offFBO;            // offscreen framebuffer object
    unsigned int offColor;          // offscreen color buffer
    unsigned int offDepthStencil;   // offscreen depth and stencil buffer

    unsigned int shadowFBO;         // shadow map framebuffer object
    unsigned int shadowTex;         // shadow map texture

    unsigned int ntexture;          // number of allocated textures
    unsigned int texture[100];      // texture names
    int textureType[100];           // type of texture (mjtTexture)

    unsigned int basePlane;         // displaylist starting positions
    unsigned int baseMesh;
    unsigned int baseHField;
    unsigned int baseBuiltin;
    unsigned int baseFontNormal;
    unsigned int baseFontBack;
    unsigned int baseFontBig;

    int     rangePlane;             // displaylist ranges
    int     rangeMesh;
    int     rangeHField;
    int     rangeBuiltin;
    int     rangeFont;

    int     charWidth[127];         // character sizes
    int     charWidthBig[127];
    int     charHeight;
    int     charHeightBig;

    int     glewInitialized;        // do not call glewInit if already set
};
typedef struct _mjrContext mjrContext;


struct _mjrOption                   // OpenGL options
{
    mjtByte stereo;                 // stereoscopic rendering
    mjtByte flags[mjNRNDFLAG];      // rendering flags (indexed by mjtRndFlag)
};
typedef struct _mjrOption mjrOption;


struct _mjrRect                     // OpenGL rectangle
{
    int left;                       // left;   default: 0
    int bottom;                     // bottom; default: 0
    int width;                      // width;  default: buffer width
    int height;                     // height; default: buffer height
};
typedef struct _mjrRect mjrRect;
