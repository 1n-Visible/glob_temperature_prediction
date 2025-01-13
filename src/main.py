
import pygame as pg
import numpy as np
#from numba import njit

import fluid

_hsv_table = [
    [0, 1, 2], [1, 0, 2], [2, 0, 1], [2, 1, 0], [1, 2, 0], [0, 2, 1]
]

def remap(array, /, new_max) -> np.ndarray:
    return array*new_max/np.max(array)

#@njit(fastmath=True)
def hsv_to_rgb(hue, saturation, value):
    h_ = (3*hue/np.pi)%6
    h = np.abs(h_%2 - 1)
    rgb = np.ones((3,)+(hue+saturation+value).shape, dtype=np.float32)
    
    h0 = (0<=h_) & (h_<1)
    h1 = (1<=h_) & (h_<2)
    h2 = (2<=h_) & (h_<3)
    h3 = (3<=h_) & (h_<4)
    h4 = (4<=h_) & (h_<5)
    h5 = (5<=h_) & (h_<6)
    X = (1-h*saturation)
    X0 = 1-saturation
    rgb[0, h1|h4] = X[h1|h4]
    rgb[1, h0|h3] = X[h0|h3]
    rgb[2, h2|h5] = X[h2|h5]
    rgb[0, h2|h3] = X0[h2|h3]
    rgb[1, h4|h5] = X0[h4|h5]
    rgb[2, h0|h1] = X0[h0|h1]
    # [value, value*(1-h*saturation), value*(1-saturation)]
    return rgb*value

def create_region(pos, diameter, size):
    region0 = np.reshape(pos, (2, 1))+np.int32([[-diameter, diameter]]*2)//2
    return tuple(slice(*np.clip(r, 0, dim)) for r, dim in zip(region0, size))

class FluidDemo:
    def __init__(self, screen_size, fps=30):
        self.size=screen_size
        self.screen=pg.display.set_mode(self.size, flags=pg.SCALED)
        pg.display.set_caption("Fluid simulation")
        
        self.fluid=fluid.FluidBox(self.size, iterations=15, density=0.5,
                                  diff=1e-5, visc=5e-5, dyes={"smoke": 0.05})
        
        self.brush_diameter=10
        self.display_image=np.zeros(self.size+(3,), dtype=np.uint8)
        self.font=pg.font.SysFont("Arial", 14, bold=True)
        self._draw_compiling()
        
        self.running=True
        self.clock=pg.time.Clock()
        self.clock.tick()
        self.fluid.step(0.1)
        self.clock.tick()
    
    def _draw_compiling(self):
        font_image=self.font.render("Compiling...", True, "#ffffff", "#000000")
        rect=font_image.get_rect()
        rect.center=self.screen.get_rect().center
        self.screen.blit(font_image, rect.topleft)
        pg.display.flip()
    
    def draw_fps(self):
        self.screen.blit(self.font.render(
            f"FPS: {self.clock.get_fps():.1f}", True, "#ffffff", "#000000"
        ), (0, 0))
    
    def update(self):
        dt=self.clock.get_time()/1000
        
        mouse_buttons = pg.mouse.get_pressed()
        mouse_pos = pg.mouse.get_pos()
        mouse_dx, mouse_dy = pg.mouse.get_rel()
        if mouse_buttons[0]:
            region=create_region(mouse_pos, self.brush_diameter, self.size)
            #print(region)
            smoke=self.fluid.get_dye("smoke")
            smoke[region]+=0.1*dt
            self.fluid.gas_density[region]+=0.1*dt
            self.fluid.vel_x[region]+=0.05*dt*mouse_dx
            self.fluid.vel_y[region]+=0.05*dt*mouse_dy
        
        keys=pg.key.get_pressed()
        if keys[pg.K_s]:
            self.fluid.vel_x*=0.95
            self.fluid.vel_y*=0.95
        if keys[pg.K_d]:
            self.fluid.gas_density*=0.95
        if keys[pg.K_LALT]:
            for i in range(3):
                self.fluid.step(dt)
        
        if self.running:
            self.fluid.gas_density*=0.995
            smoke=self.fluid.get_dye("smoke")
            smoke*=0.999
            self.fluid.step(dt)
    
    def draw(self):
        smoke=self.fluid.gas_density#get_dye("smoke")
        vx=self.fluid.vel_x
        vy=self.fluid.vel_y
        angle=np.nan_to_num(np.arctan2(vy, vx), copy=False)
        
        value=(smoke/smoke.max())**0.03
        vel=vx*vx+vy*vy
        rgb = 255*hsv_to_rgb(angle, vel/vel.max(), value)
        self.display_image[:, :, 0]=rgb[0]
        self.display_image[:, :, 1]=rgb[1]
        self.display_image[:, :, 2]=rgb[2]
        
        pg.pixelcopy.array_to_surface(self.screen, self.display_image)
        self.draw_fps()
        pg.display.flip()
    
    def mainloop(self):
        while True:
            self.update()
            for event in pg.event.get():
                if event.type==pg.QUIT:
                    return 0
                if event.type!=pg.KEYUP:
                    continue
                if event.key==pg.K_SPACE:
                    self.running = not self.running
            self.clock.tick()
            self.draw()
        return 0

def main():
    pg.init()
    pg.font.init()
    fluid_demo=FluidDemo((250, 200))
    return fluid_demo.mainloop()

if __name__=="__main__":
    main()
