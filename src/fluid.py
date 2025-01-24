'''Module implementing fluid simulation class'''

__all__=[
    "FluidBox2D"
]

import numpy as np
import numba as nb
from numba import njit

@njit(fastmath=True)
def clip_(value, max_) -> nb.float32:
    if value<0.0: return 0.0
    if value>max_: return max_
    return value

@njit()
def set_bounds(array) -> None:
    array[1:-1, 0]=array[1:-1, 1]
    array[1:-1, -1]=array[1:-1, -2]
    array[0]=array[1]
    array[-1]=array[-2]

@njit()
def set_bounds_vel(array) -> None:
    array[1:-1, 0]=0
    array[1:-1, -1]=0
    array[0]=0
    array[-1]=0

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
        x = clip_(i+vel_x[i, j], width-1)
        y = clip_(j+vel_y[i, j], height-1)
        x0=np.int32(x); y0=np.int32(y)
        dx = x-x0; dy = y-y0
        new_array[x0,   y0  ]+=value*(1-dx)*(1-dy)
        new_array[x0,   (y0+1)%height]+=value*(1-dx)*   dy
        new_array[(x0+1)%width, y0  ]+=value*   dx *(1-dy)
        new_array[(x0+1)%width, (y0+1)%height]+=value*   dx *   dy
    
    return new_array

def scalar_to_array(value, size) -> np.ndarray:
    if value is None:
        return np.zeros(size, dtype=np.float32)
    
    if isinstance(value, np.ndarray):
        if value.shape!=size:
            raise ValueError("value must be of the same shape as size")
        return np.float32(value)
    
    try:
        return np.full(size, value, dtype=np.float32)
    except TypeError:
        raise TypeError("value must be numpy array or float")

# TODO: add width*height in meters and cell size in meters
class FluidBox2D:
    def __init__(self, size, ds=0.01, iterations=10, pressure = None,
                 diff: float = 0.001, visc: float = 0.0002, dyes=None):
        self.width, self.height=self.size=size
        self.iterations=iterations
        self.ds=np.float32(ds)
        self.vel_x, self.vel_y=np.zeros((2,)+self.size, dtype=np.float32)
        
        self.pressure = (
            scalar_to_array(pressure, self.size) if pressure is not None
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
        ds=self.ds
        diff = dt*self.diff/ds**2
        #diffuse(self.pressure, diff, 15)
        dx=self.vel_x*dt/ds
        dy=self.vel_y*dt/ds
        for name, field in self.dyes.items():
            diffuse(field, diff, 12)
            self.dyes[name] = advect(field, dx, dy)
    
    def step_pressure(self, dt):
        ds=self.ds
        diffuse(self.pressure, 2e-5/ds**2, 3)
        p = self.pressure.copy()
        diffuse(p, 4e-5/ds**2, 15)
        px = (p[2:, 1:-1]-p[:-2, 1:-1])/(2*ds)
        py = (p[1:-1, 2:]-p[1:-1, :-2])/(2*ds)
        s=0.01
        pressure=self.pressure[1:-1, 1:-1]
        self.vel_x[1:-1, 1:-1] -= s*px*dt/pressure
        self.vel_y[1:-1, 1:-1] -= s*py*dt/pressure
    
    def step_inertia(self, dt):
        ds=self.ds
        visc = dt*self.visc/ds**2 # single value for all
        
        pressure = self.pressure
        inertia_x = pressure*self.vel_x
        inertia_y = pressure*self.vel_y
        diffuse(pressure, visc, self.iterations)
        diffuse(inertia_x, visc, self.iterations)
        diffuse(inertia_y, visc, self.iterations)
        
        pressure[pressure<=0.0] = 1.0
        dx = inertia_x*dt/(pressure*ds)
        dy = inertia_y*dt/(pressure*ds)
        
        self.pressure = pressure = advect(pressure, dx, dy)
        pressure[pressure<=0.0] = 1.0
        self.vel_x = advect(inertia_x, dx, dy)/pressure
        self.vel_y = advect(inertia_y, dx, dy)/pressure
        set_bounds_vel(self.vel_x)
        set_bounds_vel(self.vel_y)
        #clear_div
