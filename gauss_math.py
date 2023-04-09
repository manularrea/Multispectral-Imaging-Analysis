import rasterio
from rasterio.plot import show
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.ndimage import gaussian_filter
import math
import cv2

cmap = 'nipy_spectral'
pi=math.pi

with rasterio.open('./img/cropped/red.tif') as src:
    red=src.read(1)
with rasterio.open('./img/cropped/nir.tif') as src:
    nir = src.read(1)

with rasterio.open('./img/2. IMG_700101_000458_0092_NIR.TIF') as src:
    nir_field = src.read(1)

with rasterio.open('img/3. IMG_700101_000458_0092_RED.TIF') as src:
    red_field = src.read(1)

ndvi =(nir.astype(float)-red.astype(float)/(nir+red))
ndvi_field =(nir_field.astype(float)-red_field.astype(float)/(nir_field+red_field))
show(ndvi, cmap=cmap)

# Definir sigma_xy
def definir_sigma_xy(a,b,x):
    return np.exp(-(1/(a*(x**-b))))

#Definir Amplitud Qm
def definir_amplitud_AQm(sigma_xy):
    return (1/((2*pi)**(3/2)*(sigma_xy**2)))

#Definir patrón de dispersión Qm
def definir_dispersion_Qm(x,y,xf,yf,sigma_xy, AQm):
    return AQm*np.exp(-(xf-x)**2/(2*sigma_xy**2)-(yf-y)**2/(2*sigma_xy**2))

#Definir sigma_xyp
def definir_sigma_xyp(ks, a, b, x):
    return -ks*(1/(a*(x**(-b))))

#Definir amplitud AMCOP
def definir_amplitud_AMCOP(sigma_xyp, Qm):
    return (Qm/((2*pi)**(3/2)*(sigma_xyp**2)))

#Definir Gaussian 3D Function MCOP
def definir_MCOP(AMCOP, x,y, xp, yp, sigma_xyp):
    return AMCOP*np.exp(-(xp-x)**2/(2*sigma_xyp**2)-(yp-y)**2/(2*sigma_xyp**2))

#Variable definition

ks = 0.04 #0.4*10%
a= 1
b = -0.25 #Puede ser -0.25, -0.5, -0.75
x = red
y = nir
x_field = red_field
y_field = nir_field
xf = 0
yf = 0
xp=0
yp=0

sigma_xy = definir_sigma_xy(a,b,x)
AQm=definir_amplitud_AQm(sigma_xy)
Qm = definir_dispersion_Qm(x,y,xf,yf,sigma_xy,AQm)
sigma_xyp = definir_sigma_xyp(ks, a, b,x)
AMCOP=definir_amplitud_AMCOP(sigma_xyp, Qm)
MCOP=definir_MCOP(AMCOP, x,y, xp,yp, sigma_xyp)


# Define the range of values for Red and NIR bands
x = np.linspace(0, 1, red.shape[1])
y = np.linspace(0, 1, red.shape[0])

# Create meshgrid
X, Y = np.meshgrid(x, y)

# Compute Z values using Gaussian function
Z = MCOP

# Plot the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap=cmap)
ax.set_xlabel('Red')
ax.set_ylabel('NIR')
ax.set_zlabel('NDVI')
plt.show()