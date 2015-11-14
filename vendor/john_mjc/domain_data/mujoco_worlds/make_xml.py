import re
def do_substitution(in_lines):
    lines_iter = iter(in_lines)
    defn_lines = []
    while True:
        try:
            line = lines_iter.next()
        except StopIteration:
            raise RuntimeError("didn't find line starting with ---")
        if line.startswith('---'):
            break
        else:
            defn_lines.append(line)
    d = {}
    exec("\n".join(defn_lines), d)
    pat = re.compile("\$\((.+?)\)")
    out_lines = []
    for line in lines_iter:
        matches = pat.finditer(line)
        for m in matches:
            line = line.replace(m.group(0), str(eval(m.group(1),d)))
        out_lines.append(line)
    return out_lines





from glob import glob
import os.path as osp
infiles = glob(osp.join(osp.dirname(__file__),"*.xml.in"))
for fname in infiles:
    with open(fname,"r") as fh:
        in_lines = fh.readlines()
        out_lines = do_substitution(in_lines)
    outfname = fname[:-3]
    with open(outfname,"w") as fh:
        fh.writelines(out_lines)
