from rllab.viskit import core
import numpy as np

# MountainCar eta sweep.

eta = [0.0001, 0.0003, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
TRPO_25 = [-364.144618653, -271.245579705, -273.806627178, -335.261398865, -
           135.376743539, -113.168511452, -96.3936147988, -228.802927798, -293.397437889, -307.040316641]
TRPO_50 = [-178.39519849, -135.453463515, -162.697043188, -125.315236113, -
           120.222556566, -99.232083418, -91.5588814242, -200.969682894, -268.74659078, -284.358700469]
TRPO_75 = [-114.268140414, -119.495328787, -129.264184386, -112.943108619, -
           106.159508872, -89.4834757183, -86.5330023197, -176.516371046, -245.432734806, -262.583596362]
TRPO_25x = [-172.703486722, -101.535214952, -101.700992411, -115.848240138, -
            73.8242287852, -69.3260959225, -67.7650581158, -247.758847862, -289.040699066, -303.414278761]
TRPO_50x = [-82.4621924631, -73.5557918692, -78.8617785383, -71.3308432989, -
            70.2579702548, -66.0259755654, -66.1237087359, -221.255403933, -271.887830749, -287.526551185]
TRPO_75x = [-68.9566399634, -69.8318673694, -71.6572700503, -68.3482728886, -
            66.9613241685, -63.624547882, -64.5094868469, -193.957265361, -252.639699966, -271.75984296]
REINFORCE_25 = [-328.353673425, -342.472863661, -325.876926566, -236.107930293, -
                175.429145955, -134.054963558, -130.059675324, -138.516540292, -270.010409901, -290.655270682]
REINFORCE_50 = [-231.823602066, -212.094311282, -199.003427682, -153.314615021, -
                124.398013398, -109.421780675, -110.474181441, -129.85246266, -242.909036599, -267.199672666]
REINFORCE_75 = [-161.439311817, -137.553577481, -151.192253861, -127.926792497, -
                112.226118384, -101.125997204, -104.671286255, -121.572550185, -225.417876788, -249.815165655]
ERWR_25 = [-415.863012026, -415.483738747, -410.297235206, -404.134845351, -376.726316505, -
           127.477119453, -106.562087678, -307.738933115, -318.066881051, -317.410887916]
ERWR_50 = [-413.224188031, -406.474182387, -393.054141086, -394.172146605, -
           219.4428795, -106.939265679, -95.638718295, -285.422191793, -304.708572208, -308.297561751]
ERWR_75 = [-401.870635078, -186.015736084, -368.868006166, -379.336868628, -101.401037016, -
           96.5111701882, -88.0219523607, -266.578028061, -289.227044692, -293.747300348]

import matplotlib.pyplot as _plt
f, ax = _plt.subplots(figsize=(8, 5))
color = core.color_defaults[1 % len(core.color_defaults)]

ax.fill_between(
    eta, TRPO_25x, TRPO_75x, interpolate=True, facecolor=core.color_defaults[3 % len(core.color_defaults)], linewidth=0.0, alpha=0.3)
ax.plot(eta, TRPO_50x, color=core.color_defaults[
        3 % len(core.color_defaults)], label='TRPO')
ax.fill_between(
    eta, TRPO_25, TRPO_75, interpolate=True, facecolor=core.color_defaults[0 % len(core.color_defaults)], linewidth=0.0, alpha=0.3)
ax.plot(eta, TRPO_50, color=core.color_defaults[
        0 % len(core.color_defaults)], label='TRPO 100')
ax.fill_between(
    eta, ERWR_25, ERWR_75, interpolate=True, facecolor=core.color_defaults[2 % len(core.color_defaults)], linewidth=0.0, alpha=0.3)
ax.plot(eta, ERWR_50, color=core.color_defaults[
        2 % len(core.color_defaults)], label='ERWR')
ax.fill_between(
    eta, REINFORCE_25, REINFORCE_75, interpolate=True, facecolor=core.color_defaults[1 % len(core.color_defaults)], linewidth=0.0, alpha=0.3)
ax.plot(eta, REINFORCE_50, color=core.color_defaults[
        1 % len(core.color_defaults)], label='REINFORCE')
ax.grid(True)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
leg = ax.legend(loc='lower right', prop={'size': 12}, ncol=1)
for legobj in leg.legendHandles:
    legobj.set_linewidth(5.0)


def y_fmt(x, y):
    return str(int(np.round(x / 1000.0))) + 'k'
ax.set_xscale('log')
import matplotlib.ticker as tick
ax.set_ylim([-450, -50])

# ax.xaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
_plt.savefig('tmp' + '.pdf', bbox_inches='tight')
