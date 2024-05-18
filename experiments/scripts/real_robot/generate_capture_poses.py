import sys
import pickle as pkl
import numpy as np
import itertools

_, out_file = sys.argv

positions = []
for y, r, t in zip(np.linspace(-0.5, 0.5, 4), itertools.cycle([False, True]), [False, True, True, False]):
    xs = [-0.35, -0.2, 0.0, 0.2, 0.35]
    if t:
        xs = xs[3:]
    if r:
        xs = reversed(xs)
    for x in xs:
        positions.append((x, y))

pkl.dump((0.5, True, positions), open(out_file, 'wb'))
