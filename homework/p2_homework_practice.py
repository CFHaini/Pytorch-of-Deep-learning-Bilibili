import numpy
import matplotlib.pyplot as plt
from matplotlib import _cm
from mpl_toolkits.mplot3d import Axes3D

x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

def forward(w:numpy.ndarray,b:numpy.ndarray,x:float) -> numpy.ndarray:
    return w*x+b
def loss(y_hat:numpy.ndarray,y:float) -> numpy.ndarray:
    return (y_hat-y)**2

w_cor = numpy.arange(0.0,4.0,0.1)
b_cor = numpy.arange(-2.0,2.1,0.1)
print(f'穷举的权重数量：{len(w_cor)}')
print(f'穷举的偏置数量：{len(b_cor)}')

w,b=numpy.meshgrid(w_cor,b_cor)
mse=numpy.zeros(w.shape)

for x,y in zip(x_data, y_data):
    _y=forward(w,b,x)
    print(len(_y))
    mse=mse+loss(_y,y)    
mse= mse/(len(x_data))

print()

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d',auto_add_to_figure=False)
fig.add_axes(ax)
ax.set_xlabel('w',fontsize=20,color='cyan')
ax.set_ylabel('b',fontsize=20,color='cyan')

ax.plot_surface(w, b, mse, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
plt.show()

