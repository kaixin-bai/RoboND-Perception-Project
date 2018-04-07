#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
import pickle

def main():
    fname = '../config/training_set.sav'
    params, data = pickle.load(open(fname, 'r'))

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    for model, features in data.iteritems():
        #for f in features:
        f = np.mean(features, axis=0)
        x = np.arange(len(f))
        if model == 'soap2':
            ax1.plot(x, f)
            ax1.set_title(model)
        elif model == 'eraser':
            ax2.plot(x, f)
            ax2.set_title(model)
    plt.show()

if __name__ == "__main__":
    main()
