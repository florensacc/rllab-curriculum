import joblib
import numpy as np
from ete3 import Tree, TreeStyle
import ete3

paths = joblib.load("paths.pkl")


tree = Tree()
def populate_tree(node, paths, t):
    rew_0_paths = [p for p in paths if p["rewards"][t] == 0]
    #import pdb; pdb.set_trace()
    aaa = np.mean([p["actions"][t] for p in rew_0_paths], axis=0)
    action0 = np.argmax(np.mean([p["actions"][t] for p in rew_0_paths], axis=0))
    # import ipdb; ipdb.set_trace()
#     print(action0)
    rew_1_paths = [p for p in paths if p["rewards"][t] == 1]
    action1 = np.argmax(np.mean([p["actions"][t] for p in rew_1_paths], axis=0))
#     print(action1)

    node = node.add_child(name="[%d]" % (action0+1))

    node1 = node.add_child(name="1")
    node0 = node.add_child(name="0")
    if t + 1 < 5:
        populate_tree(node0, rew_0_paths, t+1)
        populate_tree(node1, rew_1_paths, t+1)
populate_tree(tree, paths, 0)


ts = TreeStyle()
ts.show_leaf_name = False
ts.show_scale = False

ts.rotation = 90
def my_layout(node):
    F = ete3.TextFace(node.name, tight_text=True)
    F.rotation = -90
    F.margin_right = F.margin_left = F.margin_top = F.margin_bottom = 5.0
    ete3.add_face_to_node(F, node, column=0, position="branch-right")
ts.layout_fn = my_layout
tree.show(tree_style=ts)#render(tree_style=ts)