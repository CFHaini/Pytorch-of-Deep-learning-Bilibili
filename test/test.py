import matplotlib.pyplot as plt
import numpy as np

fig= plt.figure()
ax=fig.add_subplot()
fig,ax=plt.subplots()

fig,axs=plt.subplots(2,2)
axs[0,0].plot([1,2,3,4],[1,2,3,4])
axs[0,0].set_xlabel('x')
axs[0,0].set_ylabel('y')


fig, axs = plt.subplot_mosaic([['left', 'right_top'],
                               ['left', 'right_bottom']])
axs['left'].plot([1, 2, 3], [3, 2, 1])
axs['right_top'].scatter([1, 2, 3], [1, 4, 9])
axs['right_bottom'].bar([1, 2, 3], [2, 3, 5])
plt.show()
