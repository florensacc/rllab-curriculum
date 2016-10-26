import abc


class types:
    class Type(metaclass=abc.ABCMeta):

        def __init__(self, default=None):
            self.default = default

        @abc.abstractmethod
        def validate(self, val):
            pass

        @abc.abstractmethod
        def serialize(self, val):
            pass

    class Bool(Type):

        def validate(self, val):
            assert isinstance(val, bool)
            return val

        def serialize(self, val):
            if val is True:
                return "true"
            elif val is False:
                return "false"
            else:
                raise NotImplementedError

    class SignedShort(Type):

        def validate(self, val):
            if isinstance(val, int):
                assert -2 ** 15 <= val <= 2 ** 15 - 1
                return val
            elif isinstance(val, float):
                assert abs(val - int(val)) < 1e-8
                return self.validate(int(val))
            else:
                raise ValueError("Invalid value for type SignedShort: " + str(val))

        def serialize(self, val):
            return str(int(val))

    class Float(Type):

        def validate(self, val):
            if isinstance(val, (int, float)):
                return float(val)
            else:
                raise ValueError()

        def serialize(self, val):
            return str(float(val))

    class UnsignedShort(Type):

        def validate(self, val):
            if isinstance(val, int):
                assert 0 <= val <= 2 ** 16 - 1
                return val
            elif isinstance(val, float):
                assert abs(val - int(val)) < 1e-8
                return self.validate(int(val))
            else:
                raise ValueError()

        def serialize(self, val):
            return str(int(val))

    class String8(Type):
        def validate(self, val):
            if isinstance(val, str):
                assert len(val.encode()) <= 8
                return val
            elif isinstance(val, bytes):
                assert len(val) <= 8
                return val
            else:
                raise ValueError()

        def serialize(self, val):
            return '"' + val.replace('"', '\\\"') + '"'


class Textmap(object):
    def __init__(self, namespace=None, items=None):
        self.namespace = namespace
        self.items = items

    def write(self, fio):
        print("namespace = \"%s\";" % self.namespace.replace('"', '\\"'), file=fio)
        for item in self.items:
            item.write(fio)


class Entry(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.__class__.__dict__:
                raise ValueError("Undeclared field %s set" % k)
            attr = self.__class__.__dict__[k]
            self.__dict__[k] = attr.validate(v)
        for k, attr in self.__class__.__dict__.items():
            if isinstance(attr, types.Type):
                if k not in self.__dict__:
                    self.__dict__[k] = attr.default

    def write(self, fio):
        print(self.__class__.__name__.lower(), file=fio)
        print("{", file=fio)
        for key in sorted(self.__dict__.keys()):
            val = self.__dict__[key]
            if val is not None:
                attr = self.__class__.__dict__[key]
                print("    %s = %s;" % (key, attr.serialize(val)), file=fio)
        print("}", file=fio)
        print(file=fio)


class Thing(Entry):
    id = types.UnsignedShort()
    type = types.UnsignedShort()
    x = types.SignedShort()
    y = types.SignedShort()
    z = types.SignedShort()

    single = types.Bool(default=True)
    dm = types.Bool(default=True)
    coop = types.Bool(default=True)

    skill1 = types.Bool(default=True)
    skill2 = types.Bool(default=True)
    skill3 = types.Bool(default=True)
    skill4 = types.Bool(default=True)
    skill5 = types.Bool(default=True)
    skill6 = types.Bool(default=True)
    skill7 = types.Bool(default=True)
    skill8 = types.Bool(default=True)
    class1 = types.Bool(default=True)
    class2 = types.Bool(default=True)
    class3 = types.Bool(default=True)
    class4 = types.Bool(default=True)
    class5 = types.Bool(default=True)
    class6 = types.Bool(default=True)
    class7 = types.Bool(default=True)
    class8 = types.Bool(default=True)

    scale = types.Float()
    scalex = types.Float()
    scaley = types.Float()


class Vertex(Entry):
    x = types.SignedShort()
    y = types.SignedShort()


class Linedef(Entry):
    v1 = types.UnsignedShort()
    v2 = types.UnsignedShort()
    sidefront = types.UnsignedShort()
    sideback = types.UnsignedShort()
    blocking = types.Bool()


class Sidedef(Entry):
    sector = types.UnsignedShort()
    texturemiddle = types.String8()


class Sector(Entry):
    texturefloor = types.String8()
    textureceiling = types.String8()
    heightceiling = types.SignedShort()
    lightlevel = types.SignedShort()
