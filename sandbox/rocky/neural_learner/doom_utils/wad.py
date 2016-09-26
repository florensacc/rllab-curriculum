import struct
import re
from rllab.misc import console
import os.path
import os
import subprocess
import sys
import uuid

ACC_PATH = os.environ["ACC_PATH"]


class Lump(object):
    def __init__(self, name=None, content=None):
        self.name = name
        self.content = content

    @staticmethod
    def order_key(lump):
        if re.match('E\dM\d|MAP\d\d', lump.name):
            return 0
        elif lump.name == "TEXTMAP":
            return 1
        elif lump.name == "ENDMAP":
            return 999
        else:
            return 2


class Level(object):
    def __init__(self, name=None, lumps=None):
        self.name = name
        if lumps is None:
            lumps = []
        self.lumps = lumps

    def reorganize(self):
        pass


class TextMap(object):
    pass


class WAD(object):
    def __init__(self):
        self.wad_type = None
        self.lumps = []
        self.levels = None

    @classmethod
    def from_file(cls, file_name):
        with open(file_name, "rb") as f:
            wad_type = f.read(4).decode()
            num_files = struct.unpack("<I", f.read(4))[0]
            dir_offset = struct.unpack("<I", f.read(4))[0]

            f.seek(dir_offset)

            wad = WAD()
            wad.wad_type = wad_type

            lump_infos = []
            for idx in range(num_files):
                lump_offset = struct.unpack("<I", f.read(4))[0]
                lump_size = struct.unpack("<I", f.read(4))[0]
                lump_name = f.read(8)
                lump_name = bytes([x for x in lump_name if int(x) != 0]).decode()

                lump_infos.append((lump_name, lump_offset, lump_size))

            for lump_name, lump_offset, lump_size in lump_infos:
                f.seek(lump_offset)
                lump_content = f.read(lump_size)
                wad.lumps.append(Lump(name=lump_name, content=lump_content))

        wad.reorganize()

        return wad

    def reorganize(self):
        cur_level = None
        self.levels = []
        for lump in self.lumps:
            if re.match('E\dM\d|MAP\d\d', lump.name):
                level = Level()
                level.name = lump.name
                cur_level = level
                self.levels.append(cur_level)
            else:
                cur_level.lumps.append(lump)
        if cur_level is not None and cur_level not in self.levels:
            self.levels.append(cur_level)
        for level in self.levels:
            level.reorganize()

    def save(self, file_name, force=False):
        if os.path.exists(file_name) and not force:
            if not console.query_yes_no("File at %s exists. Overwrite?" % file_name):
                sys.exit()
        with open(file_name, "wb") as f:
            f.write(self.wad_type.encode())
            f.write(struct.pack("<I", len(self.lumps)))
            # first construct string buffer for each lump
            dir_offset = 12
            for lump in self.lumps:
                lump_size = len(lump.content)
                dir_offset += lump_size

            f.write(struct.pack("<I", dir_offset))

            for lump in self.lumps:
                f.write(lump.content)

            lump_offset = 12
            # now start writing dir info

            for lump in self.lumps:
                f.write(struct.pack("<I", lump_offset))
                f.write(struct.pack("<I", len(lump.content)))
                f.write(lump.name.encode().ljust(8, b'\0'))
                lump_offset += len(lump.content)

    def save_decompressed(self, folder_name):
        console.mkdir_p(folder_name)
        self.reorganize()
        for level in self.levels:
            console.mkdir_p(os.path.join(folder_name, level.name))
            for lump in level.lumps:
                if lump.name in ["TEXTMAP", "SCRIPTS"]:
                    file_path = os.path.join(folder_name, level.name, lump.name)
                    print("Saving %s" % file_path)
                    if os.path.exists(file_path):
                        if not console.query_yes_no("File at %s exists. Overwrite?" % file_path):
                            sys.exit()
                    with open(file_path, "wb") as f:
                        f.write(lump.content)
                elif lump.name in ["ZNODES", "ENDMAP", "DIALOGUE"]:
                    pass
                else:
                    import ipdb;
                    ipdb.set_trace()

    @classmethod
    def from_folder(cls, folder_name):
        # loop through levels
        wad = WAD()
        wad.wad_type = "PWAD"
        level_names = os.listdir(folder_name)
        for level_name in sorted(level_names):
            if level_name in ["__init__.py"]:
                continue
            lumps = []
            lumps.append(Lump(name=level_name, content=b""))
            lump_names = os.listdir(os.path.join(folder_name, level_name))
            for lump_name in lump_names:
                if lump_name == "BEHAVIOR":
                    # will always recompile the BEHAVIOR node
                    assert "SCRIPTS" in lump_names
                    continue
                elif lump_name in ["__init__.py"]:
                    # skip
                    continue
                elif lump_name not in ["TEXTMAP", "TEXTMAP.py", "SCRIPTS", "ZNODES", "DIALOGUE"]:
                    import ipdb;
                    ipdb.set_trace()
                lump_file_name = os.path.join(folder_name, level_name, lump_name)

                if lump_name == "TEXTMAP.py":
                    process = subprocess.Popen(['python', lump_file_name], stdout=subprocess.PIPE)
                    lump_content, _ = process.communicate()
                    lump_name = "TEXTMAP"
                else:
                    with open(lump_file_name, "rb") as f:
                        lump_content = f.read()
                lumps.append(Lump(name=lump_name, content=lump_content))
                if lump_name == "SCRIPTS":
                    # also need to add a ACC-compiled node
                    acs_file_name = "/tmp/%s.acs" % uuid.uuid4()
                    o_file_name = "/tmp/%s.o" % uuid.uuid4()
                    with open(acs_file_name, "wb") as acs_file:
                        acs_file.write(lump_content)
                    command = [os.path.join(ACC_PATH, "acc"), "-i", ACC_PATH, acs_file_name, o_file_name]
                    subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE).communicate()
                    with open(o_file_name, "rb") as o_file:
                        lumps.append(Lump(name="BEHAVIOR", content=o_file.read()))

            lumps.append(Lump(name="ENDMAP", content=b""))
            wad.lumps.extend(sorted(lumps, key=Lump.order_key))
        wad.reorganize()
        return wad
