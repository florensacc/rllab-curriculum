import re
from scipy.stats import ttest_ind
from scipy.special import stdtr
import scipy.stats
import numpy as np
with open('data.txt') as f:
    content = f.readlines()
    mean = []
    std = []
    for line in content:
        digits = re.findall("[-+]?\d+[\.]?\d*", line)
        if len(digits) >= 15:
            if len(digits) == 15:
                if len(digits) == 15:
                    digits.append(-999)
            if len(digits) == 17:
                digits.pop(0)

            digits = np.asarray(digits).astype(np.float)
            digits[digits == -15] = -999
            mean.append(digits[::2])
            std.append(digits[1::2])
print(mean)
print(std)


def welch_t_test(mu1, s1, N1, mu2, s2, N2):
    # Construct arrays to make calculations more succint.
    N_i = np.array([N1, N2])
    dof_i = N_i - 1
    v_i = np.array([s1, s2]) ** 2
    # Calculate t-stat, degrees of freedom, use scipy to find p-value.
    t = (mu1 - mu2) / np.sqrt(np.sum(v_i / N_i))
    dof = (np.sum(v_i / N_i) ** 2) / np.sum((v_i ** 2) / ((N_i ** 2) * dof_i))
    p = scipy.stats.distributions.t.sf(np.abs(t), dof) * 2
    return t, p


def welch2(abar, avar, na, bbar, bvar, nb):
    adof = na - 1
    bdof = nb - 1
    avar = avar**2
    bvar = bvar**2
    tf = (abar - bbar) / np.sqrt(avar / na + bvar / nb)
    dof = (avar / na + bvar / nb)**2 / \
        (avar**2 / (na**2 * adof) + bvar**2 / (nb**2 * bdof))
    pf = 2 * stdtr(dof, -np.abs(tf))
    return tf, pf

counter = 0
methods = ['random', 'vpg', 'tnpg', 'rwr', 'reps', 'trpo', 'cem', 'cma-es', 'ddpg']
for m, s in zip(mean, std):
    counter += 1
    
    for i in range(len(m)):
        for j in range(i):

            result_test = welch2(m[i], s[i], 5,
                                       m[j], s[j], 5)
            if (not result_test[1] < 0.05) and (m[i] == np.max(m) or m[j] == np.max(m)) :
                print('task %s: %s not different from %s (p = %s)' % (counter, methods[i], methods[j], result_test[1]))
                print(m[i], m[j])
#             else: 
#                 print('task %s: %s IS different from %s (p = %s)' % (counter, methods[i], methods[j], result_test[1]))
    
