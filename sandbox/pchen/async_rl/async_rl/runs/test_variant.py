"""
Test whatever you want here
"""
from rllab.misc.instrument import variant, VariantGenerator

# Problem setting
vg = VariantGenerator()
vg.add("game",["frostbite","breakout"])
vg.add("a",[1,2])

for v in vg.variants():
    print(v["game"])
    print(v["a"])
