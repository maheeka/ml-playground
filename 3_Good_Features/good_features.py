# Prerequisites :
# pip install matplotlib if required.
# If you installed python with anaconda, this will already be installed
# Reference : https://www.youtube.com/watch?v=N9fDIAflCMY&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal&index=3

# How to figure out good features :
# - avoid useless features (equal like probability)
# - independent features
# - avoid redundant features
# - easy to understand

import numpy as np
import matplotlib.pyplot as pyplot

greyhounds = 500
labs = 500

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 28 + 4 * np.random.randn(labs)

pyplot.hist([grey_height, lab_height], stacked=True, color=['r','g'])
pyplot.show()
