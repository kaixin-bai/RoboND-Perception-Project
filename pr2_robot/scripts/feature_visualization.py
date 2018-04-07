#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
import pickle

def main():
    fname = '../config/training_set.sav'
    params, data = pickle.load(open(fname, 'r'))

    for model, features in data.iteritems():
        #for f in features:
        f = np.mean(features, axis=0)
        x = np.arange(len(f))
        plt.plot(x, f)
        plt.title(model)
        plt.show()

if __name__ == "__main__":
    main()
