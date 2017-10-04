//-----------------------------------//
//  This file is part of MuJoCo.     //
//  Copyright (C) 2016 Roboti LLC.   //
//-----------------------------------//


#include "mujoco.h"
#include "glfw3.h"
#include "stdlib.h"
#include "string.h"
#include <mutex>


//-------------------------------- global variables -------------------------------------

// synchronization
std::mutex gui_mutex;

// model
mjModel* m = 0;
mjData* d = 0;
char lastfile[1000] = "";

// user state
bool paused = false;
bool showoption = false;
bool showinfo = true;
bool showdepth = false;
int showhelp = 1;                   // 0: none; 1: brief; 2: full
int speedtype = 1;                  // 0: slow; 1: normal; 2: max

// abstract visualization
mjvObjects objects;
mjvCamera cam;
mjvOption vopt;
char status[1000] = "";

// OpenGL rendering
mjrContext con;
mjrOption ropt;
double scale = 1;
bool stereoavailable = false;
float depth_buffer[5120*2880];        // big enough for 5K screen
unsigned char depth_rgb[1280*720*3];  // 1/4th of screen

// selection and perturbation
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
int lastx = 0;
int lasty = 0;
int selbody = 0;
int perturb = 0;
mjtNum selpos[3] = {0, 0, 0};
mjtNum refpos[3] = {0, 0, 0};
mjtNum refquat[4] = {1, 0, 0, 0};
int needselect = 0;                 // 0: none, 1: select, 2: center 

// help strings
const char help_title[] = 
"Help\n"
"Option\n"
"Info\n"
"Depth map\n"
"Stereo\n"
"Speed\n"
"Frame\n"
"Label\n"
"Pause\n"
"Reset\n"
"Forward\n"
"Back\n"
"Forward 100\n"
"Back 100\n"
"Autoscale\n"
"Reload\n"
"Geoms\n"
"Sites\n"
"Select\n"
"Center\n"
"Zoom\n"
"Camera\n"
"Perturb\n"
"Switch Cam";

const char help_content[] = 
"F1\n"
"F2\n"
"F3\n"
"F4\n"
"F5\n"
"F6\n"
"F7\n"
"Enter\n"
"Space\n"
"BackSpace\n"
"Right arrow\n"
"Left arrow\n"
"Page Down\n"
"Page Up\n"
"Ctrl A\n"
"Ctrl L\n"
"0 - 4\n"
"Shift 0 - 4\n"
"L double-click\n"
"R double-click\n"
"Scroll or M drag\n"
"[Shift] L/R drag\n"
"Ctrl [Shift] drag\n"
"[ ]";

char opt_title[1000] = "";
char opt_content[1000];


//-------------------------------- utility functions ------------------------------------

// center and scale view
void autoscale(GLFWwindow* window)
{
    // autoscale
    cam.lookat[0] = m->stat.center[0];
    cam.lookat[1] = m->stat.center[1];
    cam.lookat[2] = m->stat.center[2];
    cam.distance = 1.5 * m->stat.extent;
    cam.camid = -1;
    cam.trackbodyid = -1;
    if( window )
    {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        mjv_updateCameraPose(&cam, (mjtNum)width/(mjtNum)height);
    }
}



// load mjb or xml model
void loadmodel(GLFWwindow* window, const char* filename, const char* xmlstring)
{
	// make sure one source is given
	if( !filename && !xmlstring )
		return;

    // load and compile
    char error[1000] = "could not load binary model";
    mjModel* mnew = 0;
	if( xmlstring )
        mnew = mj_loadXML(0, xmlstring, error, 1000);
    else if( strlen(filename)>4 && !strcmp(filename+strlen(filename)-4, ".mjb") )
        mnew = mj_loadModel(filename, 0, 0);
    else
        mnew = mj_loadXML(filename, 0, error, 1000);
    if( !mnew )
    {
        printf("%s\n", error);
        return;
    }

    // delete old model, assign new
    mj_deleteData(d);
    mj_deleteModel(m);
    m = mnew;
    d = mj_makeData(m);
    mj_forward(m, d);

    // save filename for reload
	if( !xmlstring )
	    strcpy(lastfile, filename);
	else
		lastfile[0] = 0;

    // re-create custom context
    mjr_makeContext(m, &con, 150);

    // clear perturbation state
    perturb = 0;
    selbody = 0;
    needselect = 0;

    // set title
    if( window && m->names )
        glfwSetWindowTitle(window, m->names);

    // center and scale view
    autoscale(window);
}


//--------------------------------- callbacks -------------------------------------------

// keyboard
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
    int n;

    // require model
    if( !m )
        return;

    // do not act on release
    if( act==GLFW_RELEASE )
        return;

    gui_mutex.lock();

    switch( key )
    {
    case GLFW_KEY_F1:                   // help
        showhelp++;
        if( showhelp>2 )
            showhelp = 0;
        break;

    case GLFW_KEY_F2:                   // option
        showoption = !showoption;
        break;

    case GLFW_KEY_F3:                   // info
        showinfo = !showinfo;
        break;

    case GLFW_KEY_F4:                   // depthmap
        showdepth = !showdepth;
        break;

    case GLFW_KEY_F5:                   // stereo
        if( stereoavailable )
            ropt.stereo = !ropt.stereo;
        break;

    case GLFW_KEY_F6:                   // cycle over frame rendering modes
		vopt.frame = (vopt.frame+1) % mjNFRAME;
        break;

    case GLFW_KEY_F7:                   // cycle over labeling modes
		vopt.label = (vopt.label+1) % mjNLABEL;
        break;

	case GLFW_KEY_ENTER:                // speed
        speedtype += 1;
        if( speedtype>2 )
            speedtype = 0;
        break;

    case GLFW_KEY_SPACE:                // pause
        paused = !paused;
        break;

    case GLFW_KEY_BACKSPACE:            // reset
        mj_resetData(m, d);
        mj_forward(m, d);
        break;

    case GLFW_KEY_RIGHT:                // step forward
        if( paused )
            mj_step(m, d);
        break;

    case GLFW_KEY_LEFT:                 // step back
        if( paused )
        {
            m->opt.timestep = -m->opt.timestep;
            mj_step(m, d);
            m->opt.timestep = -m->opt.timestep;
        }
        break;

    case GLFW_KEY_PAGE_DOWN:            // step forward 100
        if( paused )
            for( n=0; n<100; n++ )
                mj_step(m,d);
        break;

    case GLFW_KEY_PAGE_UP:              // step back 100
        if( paused )
        {
            m->opt.timestep = -m->opt.timestep;
            for( n=0; n<100; n++ )
                mj_step(m,d);
            m->opt.timestep = -m->opt.timestep;
        }
        break;

    case GLFW_KEY_LEFT_BRACKET:         // previous camera
        if( cam.camid>-1 )
            cam.camid--;
        break;

    case GLFW_KEY_RIGHT_BRACKET:        // next camera
        if( cam.camid<m->ncam-1 )
            cam.camid++;
        break;

    default:
        // control keys
        if( mods & GLFW_MOD_CONTROL )
        {
            if( key==GLFW_KEY_A )
                autoscale(window);
            else if( key==GLFW_KEY_L && lastfile[0] )
                loadmodel(window, lastfile, 0);

            break;
        }

        // toggle visualization flag
        for( int i=0; i<mjNVISFLAG; i++ )
            if( key==mjVISSTRING[i][2][0] )
                vopt.flags[i] = !vopt.flags[i];

        // toggle rendering flag
        for( int i=0; i<mjNRNDFLAG; i++ )
            if( key==mjRNDSTRING[i][2][0] )
                ropt.flags[i] = !ropt.flags[i];

        // toggle geom/site group
        for( int i=0; i<mjNGROUP; i++ )
            if( key==i+'0')
            {
                if( mods & GLFW_MOD_SHIFT )
                    vopt.sitegroup[i] = !vopt.sitegroup[i];
                else
                    vopt.geomgroup[i] = !vopt.geomgroup[i];
            }
    }

    gui_mutex.unlock();
}


// mouse button
void mouse_button(GLFWwindow* window, int button, int act, int mods)
{
    // past data for double-click detection
    static int lastbutton = 0;
    static double lastclicktm = 0;

    // update button state
    button_left =   (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
    button_right =  (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

    // update mouse position
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    lastx = (int)(scale*x);
    lasty = (int)(scale*y);

    // require model
    if( !m )
        return;

    gui_mutex.lock();

    // set perturbation
    int newperturb = 0;
    if( (mods & GLFW_MOD_CONTROL) && selbody>0 )
    {
        // right: translate;  left: rotate
        if( button_right )
            newperturb = mjPERT_TRANSLATE;
        else if( button_left )
            newperturb = mjPERT_ROTATE;

        // perturbation onset: reset reference
        if( newperturb && !perturb )
        {
            int id = paused ? m->body_rootid[selbody] : selbody;
            mju_copy3(refpos, d->xpos+3*id);
            mju_copy(refquat, d->xquat+4*id, 4);
        }
    }
    perturb = newperturb;

    // detect double-click (250 msec)
    if( act==GLFW_PRESS && glfwGetTime()-lastclicktm<0.25 && button==lastbutton )
    {
        if( button==GLFW_MOUSE_BUTTON_LEFT )
            needselect = 1;
        else
            needselect = 2;

        // stop perturbation on select
        perturb = 0;
    }

    // save info
    if( act==GLFW_PRESS )
    {
        lastbutton = button;
        lastclicktm = glfwGetTime();
    }

    gui_mutex.unlock();
}


// mouse move
void mouse_move(GLFWwindow* window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if( !button_left && !button_middle && !button_right )
        return;

    // compute mouse displacement, save
    float dx = (int)(scale*xpos) - (float)lastx;
    float dy = (int)(scale*ypos) - (float)lasty;
    lastx = (int)(scale*xpos);
    lasty = (int)(scale*ypos);

    // require model
    if( !m )
        return;

    // get current window size
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if( button_right )
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if( button_left )
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    gui_mutex.lock();

    // perturbation
    if( perturb )
    {
        if( selbody>0 )
            mjv_moveObject(action, dx, dy, &cam.pose, 
                           (float)width, (float)height, refpos, refquat);
    }

    // camera control
    else
        mjv_moveCamera(action, dx, dy, &cam, (float)width, (float)height);

    gui_mutex.unlock();
}


// scroll
void scroll(GLFWwindow* window, double xoffset, double yoffset)
{
    // require model
    if( !m )
        return;

    // get current window size
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    // scroll
    gui_mutex.lock();
    mjv_moveCamera(mjMOUSE_ZOOM, 0, (float)(-20*yoffset), &cam, (float)width, (float)height);
    gui_mutex.unlock();
}


// drop
void drop(GLFWwindow* window, int count, const char** paths)
{
    // make sure list is non-empty
    if( count>0 )
    {
        gui_mutex.lock();
        loadmodel(window, paths[0], 0);
        gui_mutex.unlock();
    }
}


//-------------------------------- simulation and rendering -----------------------------

// make option string
void makeoptionstring(const char* name, char key, char* buf)
{
    int i=0, cnt=0;

    // copy non-& characters
    while( name[i] && i<50 )
    {
        if( name[i]!='&' )
            buf[cnt++] = name[i];

        i++;
    }

    // finish
    buf[cnt] = ' ';
    buf[cnt+1] = '(';
    buf[cnt+2] = key;
    buf[cnt+3] = ')';
    buf[cnt+4] = 0;
}


// advance simulation
void advance(void)
{
    // perturbations
    if( selbody>0 )
    {
        // fixed object: edit
        if( m->body_jntnum[selbody]==0 && m->body_parentid[selbody]==0 )
            mjv_mouseEdit(m, d, selbody, perturb, refpos, refquat);
    
        // movable object: set mouse perturbation
        else
            mjv_mousePerturb(m, d, selbody, perturb, refpos, refquat, 
                             d->xfrc_applied+6*selbody);
    }

    // advance simulation
	mj_step(m, d);

    // clear perturbation
    if( selbody>0 )
        mju_zero(d->xfrc_applied+6*selbody, 6);
}


// render
void render(GLFWwindow* window)
{
    // past data for FPS calculation
    static double lastrendertm = 0;

    // get current window rectangle
    mjrRect rect = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &rect.width, &rect.height);

    double duration = 0;
    gui_mutex.lock();

    // no model: empty screen
    if( !m )
    {
		mjr_rectangle(rect, 0, 0, rect.width, rect.height, 0.2, 0.3, 0.4, 1);
		mjr_overlay(rect, mjGRID_TOPLEFT, 0, "Drag-and-drop model file here", 0, &con);
        gui_mutex.unlock();
        return;
    }

    // start timers
    double starttm = glfwGetTime();
    mjtNum startsimtm = d->time;

    // paused
    if( paused )
    {
        // edit
        mjv_mouseEdit(m, d, selbody, perturb, refpos, refquat);

        // recompute to refresh rendering
        mj_forward(m, d);

        // 15 msec delay
        while( glfwGetTime()-starttm<0.015 );
    }

    // running
    else
    {
        // simulate for 15 msec of CPU time
        int n = 0;
        while( glfwGetTime()-starttm<0.015 )
        {
            // step at specified speed
            if( (speedtype==0 && n==0) || (speedtype==1 && d->time-startsimtm<0.016) || speedtype==2 )
            {
                advance();
                n++;
            }

            // simulation already done: compute duration
            else if( duration==0 && n )
                duration = 1000*(glfwGetTime() - starttm)/n;

        }

        // compute duration if not already computed
        if( duration==0 && n )
            duration = 1000*(glfwGetTime() - starttm)/n;
    }

    // update simulation statistics
    if( !paused )
        sprintf(status, "%.1f\n%d (%d)\n%.2f\n%.0f          \n%.2f\n%.2f (%2.0f it)\n%d\n%d\n%d",
                d->time, d->nefc, d->ncon, 
                duration, 1.0/(glfwGetTime()-lastrendertm),
                d->energy[0]+d->energy[1],
                mju_log10(mju_max(mjMINVAL, d->solverstat[1])),
				d->solverstat[0],
				cam.camid, vopt.frame, vopt.label );
    lastrendertm = glfwGetTime();

    // create geoms and lights
    mjv_makeGeoms(m, d, &objects, &vopt, mjCAT_ALL, selbody, 
                  (perturb & mjPERT_TRANSLATE) ? refpos : 0, 
                  (perturb & mjPERT_ROTATE) ? refquat : 0, selpos); 
    mjv_makeLights(m, d, &objects);

    // update camera
    mjv_setCamera(m, d, &cam);
    mjv_updateCameraPose(&cam, (mjtNum)rect.width/(mjtNum)rect.height);

    // selection
    if( needselect )
    {
        // find selected geom
        mjtNum pos[3];
        int selgeom = mjr_select(rect, &objects, lastx, rect.height - lasty, 
                                 pos, 0, &ropt, &cam.pose, &con);

        // set lookat point
        if( needselect==2 )
        {
            if( selgeom >= 0 )
                mju_copy3(cam.lookat, pos);
        }

        // set body selection
        else
        {
            if( selgeom>=0 && objects.geoms[selgeom].objtype==mjOBJ_GEOM )
            {
                // record selection
                selbody = m->geom_bodyid[objects.geoms[selgeom].objid];

                // clear if invalid
                if( selbody<0 || selbody>=m->nbody )
                    selbody = 0;

                // otherwise compute selpos
                else
                {
                    mjtNum tmp[3];
                    mju_sub3(tmp, pos, d->xpos+3*selbody);
                    mju_mulMatTVec(selpos, d->xmat+9*selbody, tmp, 3, 3);
                }
            }
            else
                selbody = 0;
        }

        needselect = 0;
    }

    // render rgb
    mjr_render(0, rect, &objects, &ropt, &cam.pose, &con);

    // show depth map
    if( showdepth )
    {
        // get the depth buffer
        mjr_getBackbuffer(0, depth_buffer, rect, &con);  // not working with 5k ???

        // convert to RGB, subsample by 4
        for( int r=0; r<rect.height; r+=4 )
            for( int c=0; c<rect.width; c+=4 )
            {
                // get subsampled address
                int adr = (r/4)*(rect.width/4) + c/4;

                // assign rgb
                depth_rgb[3*adr] = depth_rgb[3*adr+1] = depth_rgb[3*adr+2] = 
                    (unsigned char)((1.0f-depth_buffer[r*rect.width+c])*255.0f);
            }

        // show in bottom-right corner
        mjr_showBuffer(depth_rgb, rect.width/4, rect.height/4, (3*rect.width)/4, 0, &con);
    }

    // show overlays
    if( showhelp==1 )
        mjr_overlay(rect, mjGRID_TOPLEFT, 0, "Help  ", "F1  ", &con);
    else if( showhelp==2 )
        mjr_overlay(rect, mjGRID_TOPLEFT, 0, help_title, help_content, &con);

    if( showinfo )
    {
        if( paused )
            mjr_overlay(rect, mjGRID_BOTTOMLEFT, 0, "PAUSED", 0, &con);
        else
            mjr_overlay(rect, mjGRID_BOTTOMLEFT, 0, 
                "Time\nSize\nCPU\nFPS\nEngy\nStat\nCam\nFrame\nLabel", status, &con);
    }

    if( showoption )
    {
        int i;
        char buf[100];

        // fill titles on first pass
        if( !opt_title[0] )
        {
            for( i=0; i<mjNRNDFLAG; i++)
            {
                makeoptionstring(mjRNDSTRING[i][0], mjRNDSTRING[i][2][0], buf);
                strcat(opt_title, buf);
                strcat(opt_title, "\n");
            }
            for( i=0; i<mjNVISFLAG; i++)
            {
                makeoptionstring(mjVISSTRING[i][0], mjVISSTRING[i][2][0], buf);
                strcat(opt_title, buf);
                if( i<mjNVISFLAG-1 )
                    strcat(opt_title, "\n");
            }
        }

        // fill content
        opt_content[0] = 0;
        for( i=0; i<mjNRNDFLAG; i++)
        {
            strcat(opt_content, ropt.flags[i] ? " + " : "   ");
            strcat(opt_content, "\n");
        }
        for( i=0; i<mjNVISFLAG; i++)
        {
            strcat(opt_content, vopt.flags[i] ? " + " : "   ");
            if( i<mjNVISFLAG-1 )
                strcat(opt_content, "\n");
        }

        // show
        mjr_overlay(rect, mjGRID_TOPRIGHT, 0, opt_title, opt_content, &con);
    }

    gui_mutex.unlock();
}



//-------------------------------- main function ----------------------------------------

int main(int argc, const char** argv)
{
	// print version, check compatibility
	printf("MuJoCo Pro library version %.2lf\n\n", 0.01*mj_version());
	if( mjVERSION_HEADER!=mj_version() )
		mju_error("Headers and library have different versions");

	// activate MuJoCo license
	mj_activate("mjkey.txt");
	
    // init GLFW, set multisampling
    if (!glfwInit())
        return 1;
    glfwWindowHint(GLFW_SAMPLES, 4);

    // try stereo if refresh rate is at least 100Hz
    GLFWwindow* window = 0;
    if( glfwGetVideoMode(glfwGetPrimaryMonitor())->refreshRate>=100 )
    {
        glfwWindowHint(GLFW_STEREO, 1);
        window = glfwCreateWindow(1200, 900, "Simulate", NULL, NULL);
        if( window )
            stereoavailable = true;
    }

    // no stereo: try mono
    if( !window )
    {
        glfwWindowHint(GLFW_STEREO, 0);
        window = glfwCreateWindow(1200, 900, "Simulate", NULL, NULL);
    }
    if( !window )
    {
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);

    // determine retina scaling
    int width, width1, height;
    glfwGetFramebufferSize(window, &width, &height);
    glfwGetWindowSize(window, &width1, &height);
    scale = (double)width/(double)width1;

    // init MuJoCo rendering
    mjv_makeObjects(&objects, 1000);
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&vopt);
    mjr_defaultOption(&ropt);
    mjr_defaultContext(&con);
    mjr_makeContext(m, &con, 200);

    // load model if filename given as argument
    if( argc==2 )
        loadmodel(window, argv[1], 0);

    // set GLFW callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);
    glfwSetDropCallback(window, drop);

    // main loop
    while( !glfwWindowShouldClose(window) )
    {
        // simulate and render
        render(window);

        // finalize
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // delete everything we allocated
    mj_deleteData(d);
    mj_deleteModel(m);
    mjr_freeContext(&con);
    mjv_freeObjects(&objects);

    // terminate
    glfwTerminate();
	mj_deactivate();
    return 0;
}
