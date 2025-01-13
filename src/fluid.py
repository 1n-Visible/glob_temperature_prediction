'''Module implementing fluid simulation class'''

__all__=[
    "FluidBox"
]

import numpy as np
import numba as nb
from numba import njit

@njit(fastmath=True)
def clip_(value, max_) -> nb.float32:
    if value<1.0: return 1.0
    if value>max_: return max_
    return value

@njit()
def set_bounds(array) -> None:
    array[1:-1, 0]=array[1:-1, 1]
    array[1:-1, -1]=array[1:-1, -2]
    array[0]=array[1]
    array[-1]=array[-2]

@njit()
def set_bounds_vel(vel, vel_type) -> None:
    ...

@njit()
def diffuse(array, diff, iterations) -> None:
    set_bounds(array)
    array0=array.copy()
    for k in range(iterations):
        array[1:-1, 1:-1]=(array0[1:-1, 1:-1] + diff*(
            array[1:-1, 2:]+array[1:-1, :-2]+array[2:, 1:-1]+array[:-2, 1:-1]
        ))/(1+4*diff)
        set_bounds(array)

#TODO: add new_array to save from extra allocations
@njit(looplift=True, fastmath=True, boundscheck=True)
def advect(array0, vel_x, vel_y) -> np.ndarray:
    width, height = size = np.shape(array0)
    new_array=np.zeros(size, dtype=np.float32)
    
    for (i, j), value in np.ndenumerate(array0[1:-1, 1:-1]):
        i+=1; j+=1
        x = clip_(i+vel_x[i, j], width-2)
        y = clip_(j+vel_y[i, j], height-2)
        x0=np.int32(x); y0=np.int32(y)
        dx = x-x0; dy = y-y0
        new_array[x0,   y0  ]+=value*(1-dx)*(1-dy)
        new_array[x0,   y0+1]+=value*(1-dx)*   dy
        new_array[x0+1, y0  ]+=value*   dx *(1-dy)
        new_array[x0+1, y0+1]+=value*   dx *   dy
    
    return new_array

def scalar_to_array(value, size) -> np.ndarray:
    if value is None:
        return np.zeros(size, dtype=np.float32)
    
    if isinstance(value, np.ndarray):
        if value.shape!=size:
            raise ValueError("value must be of the same shape as size")
        return np.float32(density)
    
    try:
        return np.full(size, value, dtype=np.float32)
    except TypeError:
        raise TypeError("value must be numpy array or float")

# TODO: add width*height in meters and cell size in meters
class FluidBox:
    def __init__(self, size, ds=0.01, iterations=10, density = None,
                 diff: float = 0.001, visc: float = 0.0002, dyes=None):
        self.width, self.height=self.size=size
        self.ds=np.float32(ds)
        self.vel_x, self.vel_y=np.zeros((2,)+self.size, dtype=np.float32)
        
        self.gas_density = (
            scalar_to_array(density, self.size) if density is not None
            else np.ones(size, dtype=np.float32)
        )
        
        self.diff=np.float32(diff)
        self.visc=np.float32(visc)
        
        self.dyes=dyes or {}
        for name, field in self.dyes.items():
            self.dyes[name]=scalar_to_array(field, self.size)
    
    def get_dye(self, name):
        return self.dyes[name]
    
    def step(self, dt):
        self.step_pressure(dt)
        self.step_density(dt)
        self.step_inertia(dt)
    
    def step_density(self, dt):
        diff = dt*self.diff/self.ds**2
        #diffuse(self.gas_density, diff, 15)
        for name, field in self.dyes.items():
            diffuse(field, diff, 12)
            self.dyes[name] = advect(field, self.vel_x, self.vel_y)
    
    def step_pressure(self, dt):
        diffuse(self.gas_density, 1.0, 3)
        p = self.gas_density.copy()
        diffuse(p, 5.4, 15)
        px = p[2:, 1:-1]-p[:-2, 1:-1]
        py = p[1:-1, 2:]-p[1:-1, :-2]
        
        density=self.gas_density[1:-1, 1:-1]
        s=0.9
        self.vel_x[1:-1, 1:-1] -= s*px*dt/density
        self.vel_y[1:-1, 1:-1] -= s*py*dt/density
    
    def step_inertia(self, dt):
        visc = dt*self.visc/self.ds**2 # single value for all
        
        density = self.gas_density
        inertia_x = density*self.vel_x
        inertia_y = density*self.vel_y
        diffuse(density, visc, 15)
        diffuse(inertia_x, visc, 15)
        diffuse(inertia_y, visc, 15)
        
        density[density<=0.0] = 1.0
        vel = inertia_x/density, inertia_y/density
        
        self.gas_density = density = advect(density, *vel)
        density[density<=0.0] = 1.0
        self.vel_x = advect(inertia_x, *vel)/density
        self.vel_y = advect(inertia_y, *vel)/density
        #clear_div
