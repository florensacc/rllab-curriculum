# pylint: disable=no-init, too-few-public-methods, old-style-class

import xml.etree.ElementTree as ET
from rllab.mdp.box2d.parser.xml_types import XmlElem, XmlChild, XmlAttr, \
    XmlChildren
from rllab.mdp.box2d.parser.xml_attr_types import Tuple, Float, Choice, \
    String, List, Point2D, Hex, Int, Angle, Bool, Either
import Box2D


class XmlBox2D(XmlElem):

    tag = "box2d"

    class Meta:
        world = XmlChild("world", lambda: XmlWorld, required=True)

    def __init__(self):
        self.world = None

    def to_box2d(self, extra_data):
        return self.world.to_box2d(extra_data)


class XmlWorld(XmlElem):

    tag = "world"

    class Meta:
        bodies = XmlChildren("body", lambda: XmlBody)
        gravity = XmlAttr("gravity", Point2D())
        joints = XmlChildren("joint", lambda: XmlJoint)
        states = XmlChildren("state", lambda: XmlState)
        controls = XmlChildren("control", lambda: XmlControl)
        warmStarting = XmlAttr("warmstart", Bool())
        continuousPhysics = XmlAttr("continuous", Bool())
        subStepping = XmlAttr("substepping", Bool())
        velocityIterations = XmlAttr("velitr", Int())
        positionIterations = XmlAttr("positr", Int())
        timeStep = XmlAttr("timestep", Float())

    def __init__(self):
        self.bodies = []
        self.gravity = None
        self.joints = []
        self.states = []
        self.controls = []
        self.warmStarting = True
        self.continuousPhysics = True
        self.subStepping = False
        self.velocityIterations = 8
        self.positionIterations = 3
        self.timeStep = 0.02

    def to_box2d(self, extra_data):
        world = Box2D.b2World(allow_sleeping=False)
        world.warmStarting = self.warmStarting
        world.continuousPhysics = self.continuousPhysics
        world.subStepping = self.subStepping
        extra_data.velocityIterations = self.velocityIterations
        extra_data.positionIterations = self.positionIterations
        extra_data.timeStep = self.timeStep
        if self.gravity:
            world.gravity = self.gravity
        for body in self.bodies:
            body.to_box2d(world, self, extra_data)
        for joint in self.joints:
            joint.to_box2d(world, self, extra_data)
        for state in self.states:
            state.to_box2d(world, self, extra_data)
        for control in self.controls:
            control.to_box2d(world, self, extra_data)
        return world


class XmlBody(XmlElem):

    tag = "body"

    TYPES = ["static", "kinematic", "dynamic"]

    class Meta:
        color = XmlAttr("color", List(Float()))
        name = XmlAttr("name", String())
        typ = XmlAttr("type", Choice("static", "kinematic", "dynamic"),
                      required=True)
        fixtures = XmlChildren("fixture", lambda: XmlFixture)
        position = XmlAttr("position", Point2D())

    def __init__(self):
        self.color = None
        self.name = None
        self.typ = None
        self.position = None
        self.fixtures = []

    def to_box2d(self, world, xml_world, extra_data):
        body = world.CreateBody(type=self.TYPES.index(self.typ))
        body.userData = dict(
            name=self.name,
            color=self.color,
        )
        if self.position:
            body.position = self.position
        for fixture in self.fixtures:
            fixture.to_box2d(body, self, extra_data)
        return body


class XmlFixture(XmlElem):

    tag = "fixture"

    class Meta:
        shape = XmlAttr("shape",
                        Choice("polygon", "circle", "edge"), required=True)
        vertices = XmlAttr("vertices", List(Point2D()))
        box = XmlAttr("box", Either(
            Point2D(),
            Tuple(Float(), Float(), Point2D(), Angle())))
        radius = XmlAttr("radius", Float())
        width = XmlAttr("width", Float())
        center = XmlAttr("center", Point2D())
        angle = XmlAttr("angle", Angle())
        position = XmlAttr("position", Point2D())
        friction = XmlAttr("friction", Float())
        density = XmlAttr("density", Float())
        category_bits = XmlAttr("category_bits", Hex())
        mask_bits = XmlAttr("mask_bits", Hex())
        group = XmlAttr("group", Int())

    def __init__(self):
        self.shape = None
        self.vertices = None
        self.box = None
        self.friction = None
        self.density = None
        self.category_bits = None
        self.mask_bits = None
        self.group = None
        self.radius = None
        self.width = None
        self.center = None
        self.angle = None

    def to_box2d(self, body, xml_body, extra_data):
        attrs = dict()
        if self.friction:
            attrs["friction"] = self.friction
        if self.density:
            attrs["density"] = self.density
        if self.group:
            attrs["groupIndex"] = self.group
        if self.radius:
            attrs["radius"] = self.radius
        if self.shape == "polygon":
            if self.box:
                fixture = body.CreatePolygonFixture(
                    box=self.box, **attrs)
            else:
                fixture = body.CreatePolygonFixture(
                    vertices=self.vertices, **attrs)
        elif self.shape == "edge":
            fixture = body.CreateEdgeFixture(vertices=self.vertices, **attrs)
        elif self.shape == "circle":
            if self.center:
                attrs["pos"] = self.center
            fixture = body.CreateCircleFixture(**attrs)
        else:
            assert False
        return fixture


def _get_name(x):
    if isinstance(x.userData, dict):
        return x.userData.get('name')
    return None


def find_body(world, name):
    return [body for body in world.bodies if _get_name(body) == name][0]


def find_joint(world, name):
    return [joint for joint in world.joints if _get_name(joint) == name][0]


class XmlJoint(XmlElem):

    tag = "joint"

    JOINT_TYPES = {
        "revolute": Box2D.b2RevoluteJoint,
        "friction": Box2D.b2FrictionJoint,
    }

    class Meta:
        bodyA = XmlAttr("bodyA", String(), required=True)
        bodyB = XmlAttr("bodyB", String(), required=True)
        anchor = XmlAttr("anchor", Tuple(Float(), Float()))
        limit = XmlAttr("limit", Tuple(Angle(), Angle()))
        ctrllimit = XmlAttr("ctrllimit", Tuple(Angle(), Angle()))
        typ = XmlAttr("type", Choice("revolute", "friction"), required=True)
        name = XmlAttr("name", String())
        motor = XmlAttr("motor", Bool())

    def __init__(self):
        self.bodyA = None
        self.bodyB = None
        self.anchor = None
        self.limit = None
        self.ctrllimit = None
        self.motor = False
        self.typ = None
        self.name = None

    def to_box2d(self, world, xml_world, extra_data):
        bodyA = find_body(world, self.bodyA)
        bodyB = find_body(world, self.bodyB)
        args = dict()
        if self.typ == "revolute":
            if self.anchor:
                args["anchor"] = self.anchor
            if self.limit:
                args["enableLimit"] = True
                args["lowerAngle"] = self.limit[0]
                args["upperAngle"] = self.limit[1]
        elif self.typ == "friction":
            if self.anchor:
                args["anchor"] = self.anchor
        else:
            raise NotImplementedError
        userData = dict(
            ctrllimit=self.ctrllimit,
            motor=self.motor,
            name=self.name
        )
        joint = world.CreateJoint(type=self.JOINT_TYPES[self.typ],
                                  bodyA=bodyA,
                                  bodyB=bodyB,
                                  **args)
        joint.userData = userData
        return joint


class XmlState(XmlElem):

    tag = "state"

    class Meta:
        typ = XmlAttr(
            "type", Choice(
                "xpos", "ypos", "xvel", "yvel", "apos", "avel"))
        body = XmlAttr("body", String())

    def __init__(self):
        self.typ = None
        self.body = None

    def to_box2d(self, world, xml_world, extra_data):
        extra_data.states.append(self)


class XmlControl(XmlElem):

    tag = "control"

    class Meta:
        typ = XmlAttr("type", Choice("force", "torque"), required=True)
        body = XmlAttr(
            "body", String(),
            help="name of the body to apply force on")
        joint = XmlAttr(
            "joint", String(),
            help="name of the joint")
        anchor = XmlAttr(
            "anchor", Point2D(),
            help="location of the force in local coordinate frame")
        direction = XmlAttr(
            "direction", Point2D(),
            help="direction of the force in local coordinate frame")
        ctrllimit = XmlAttr(
            "ctrllimit", Tuple(Float(), Float()),
            help="limit of the control input in Newton")

    def __init__(self):
        self.typ = None
        self.body = None
        self.joint = None
        self.anchor = None
        self.direction = None
        self.ctrllimit = None

    def to_box2d(self, world, xml_world, extra_data):
        extra_data.controls.append(self)


class ExtraData(object):

    def __init__(self):
        self.states = []
        self.controls = []
        self.velocityIterations = None
        self.positionIterations = None
        self.timeStep = None


def world_from_xml(s):
    extra_data = ExtraData()
    box2d = XmlBox2D.from_xml(ET.fromstring(s))
    world = box2d.to_box2d(extra_data)
    return world, extra_data
