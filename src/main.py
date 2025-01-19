import pygame as pg
import numpy as np
import netCDF4
import json

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from fluid import *
from utils import *

def label_update_mouse(self, event):
    self.mx=event.x()
    self.my=event.y()

class SimulationApp(QApplication):
    MINSIZE=(300, 500)
    EARTH_RAD=6378
    
    def __init__(self, argv):
        super().__init__(argv)
        self.win = QMainWindow()
        self.main = QWidget()
        self.win.setCentralWidget(self.main)
        
        self.init_events()
        self.init_simulation()
        self.init_ui()
        self.win.setMinimumSize(*self.MINSIZE)
        self.win.setWindowTitle("Simulation Window")
        self.win.show()
    
    def init_ui(self):
        self.layout = QGridLayout()
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(40)
        
        self.init_control_panel()
        
        self.map_image=QPixmap("data/map.png").scaled(*self.img_size)
        self.img_label=QLabel()
        self.map_label=QLabel()
        #self.map_label.setFrameStyle(QFrame.Sunken)
        self.map_label.setPixmap(self.map_image)
        self.map_label.setMouseTracking(True)
        self.map_label.mouseMoveEvent=label_update_mouse
        
        self.layout.addLayout(self.params, 0, 0)
        self.layout.addWidget(self.map_label, 0, 1)
        self.layout.addWidget(self.img_label, 0, 1)
        self.main.setLayout(self.layout)
    
    def init_control_panel(self):
        self.params = QVBoxLayout()
        
        self.dt_label = QLabel("Step size: ")
        self.dt_slider = QSlider(Qt.Horizontal)
        self.dt_slider.valueChanged.connect(self.set_fluid_dt)
        self.dt_slider.setRange(1, 50)
        self.speed_label = QLabel("Simulation speed:")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.valueChanged.connect(self.set_fluid_speed)
        self.speed_slider.setRange(1, 100)
        
        self.blast = QGridLayout()
        self.lat_box = QLineEdit("48.0")
        self.lon_box = QLineEdit("32.0")
        self.pressure_box = QLineEdit("10.0")
        self.emissions_box = QLineEdit("1500.0")
        self.lat_box.setMaxLength(8); self.lon_box.setMaxLength(8)
        self.blast.addWidget(QLabel("Широта:"), 0, 0)
        self.blast.addWidget(QLabel("Довгота:"), 1, 0)
        self.blast.addWidget(QLabel("Енергія вибуху (МДж):"), 2, 0) # J = Pa*m**3
        self.blast.addWidget(QLabel("Об'єм викидів (л):"), 3, 0)
        self.blast.addWidget(self.lat_box, 0, 1)
        self.blast.addWidget(self.lon_box, 1, 1)
        self.blast.addWidget(self.pressure_box, 2, 1)
        self.blast.addWidget(self.emissions_box, 3, 1)
        
        self.gen_button = QPushButton("Вибух")
        self.gen_button.clicked.connect(self.create_blast)
        self.pause_button = QPushButton("Пауза") # QToolButton for icon
        self.pause_button.clicked.connect(self.play_pause)
        self.info_label = QLabel()
        self.info_label.mx=0; self.info_label.my=0
        
        self.params.addWidget(self.dt_label)
        self.params.addWidget(self.dt_slider)
        self.params.addWidget(self.speed_label)
        self.params.addWidget(self.speed_slider)
        self.params.addLayout(self.blast)
        self.params.addStretch()
        self.params.addWidget(self.gen_button, alignment=Qt.AlignHCenter)
        self.params.addWidget(self.pause_button)
        self.params.addWidget(self.info_label)
    
    def init_events(self):
        self.fluid_timer = QTimer(self)
        self.fluid_timer.setSingleShot(False)
        self.fluid_timer.setInterval(10) # in milliseconds
        self.fluid_timer.timeout.connect(self.update_fluid_image)
        self.fluid_timer.start()
    
    def init_simulation(self):
        with open("data/dimensions.json", "r") as file:
            dim_data=json.load(file)
        self.min_lat, self.max_lat = dim_data["lat"]
        self.min_lon, self.max_lon = dim_data["lon"]
        
        self.width_deg = self.max_lon-self.min_lon
        self.height_deg = self.max_lat-self.min_lat
        self.width_km = np.pi*self.EARTH_RAD*self.width_deg/180
        self.height_km = np.pi*self.EARTH_RAD*self.height_deg/180
        #print(self.width_km, self.height_km)
        self.img_width=500; self.ds=self.width_km/self.img_width
        self.img_height=int(self.height_km/self.ds)
        
        self.img_size=(self.img_width, self.img_height)
        self.fluid=FluidBox2D(self.img_size, ds=0.01, iterations=10,
                              density=None, diff=1e-5, visc=2e-4,
                              dyes={"chemical": 0.0})
        self.img_array=np.zeros(self.img_size+(4,), dtype=np.uint8)
        self.img_array[:, :, 0]=80
        self.img_array[:, :, 1]=230
        self.img_array[:, :, 2]=80
        
        with netCDF4.Dataset("data/u-wind.nc", "r") as file:
            u_wind=file["uwnd"][-1].transpose(1, 0) # отримати дані за сьогодні
        with netCDF4.Dataset("data/v-wind.nc", "r") as file:
            v_wind=file["vwnd"][-1].transpose(1, 0)
        with netCDF4.Dataset("data/pressure.nc", "r") as file:
            slp=file["slp"][-1].transpose(1, 0)
        
        lon_min=self.min_lon/360; lon_max=self.max_lon/360
        lat_min=(90-self.max_lat)/180; lat_max=(90-self.min_lat)/180
        scale=np.reshape(
            np.array([lon_max-lon_min, lat_max-lat_min])/self.img_size,
            (2, 1, 1)
        )
        pos=tuple(np.int32(np.reshape(u_wind.shape, (2, 1, 1))*(
            np.mgrid[:self.img_width, :self.img_height]*scale +
            np.reshape([lon_min, lat_min], (2, 1, 1))
        )))
        self.fluid.vel_x[:]=v_wind[pos]/100 # from m/s to km/s
        self.fluid.vel_y[:]=u_wind[pos]/100
        self.fluid.pressure[:]=slp[pos]
        
        self.is_running=False
        self.fluid_dt=0.1
        self.fluid_speed=1.0
    
    def update_fluid_image(self):
        self.info_label.setText(f"({self.info_label.mx}, {self.info_label.my})")
        
        if self.is_running:
            self.fluid.step(self.fluid_dt)
        
        #vx=self.fluid.vel_x
        #vy=self.fluid.vel_y
        #angle=np.nan_to_num(np.arctan2(vy, vx), copy=False)
        chemical=self.fluid.get_dye("chemical")
        #vel=vx*vx+vy*vy
        #rgb = 255*hsv_to_rgb(angle, 1.0, 1.0)
        #self.img_array[:, :, 0]=rgb[0]
        #self.img_array[:, :, 1]=rgb[1]
        #self.img_array[:, :, 2]=rgb[2]
        self.img_array[:, :, 3]=255*chemical/chemical.max()
        arr1=np.require(self.img_array.transpose(1, 0, 2), requirements="C")
        self.qimage=QImage(
            arr1, self.img_width, self.img_height,
            QImage.Format_RGBA8888
        )
        self.img_label.setPixmap(QPixmap(self.qimage))
    
    def set_blast(self):
        self.blast_lat=float(self.lat_box.text())
        self.blast_lon=float(self.lon_box.text())
        self.blast_pressure=float(self.pressure_box.text())
        self.blast_emissions=float(self.emissions_box.text())
    
    def create_blast(self):
        try:
            self.set_blast()
        except (TypeError, ValueError):
            return
        
        expl_size = 10
        uv_pos = np.array([
            1-(self.blast_lat-self.min_lat)/self.height_deg,
            (self.blast_lon-self.min_lon)/self.width_deg
        ])
        region=create_region(np.int32(uv_pos*self.img_size), expl_size, self.img_size)
        self.fluid.pressure[region]+=self.blast_pressure/expl_size**2
        chemical=self.fluid.get_dye("chemical")
        chemical[region]+=self.blast_emissions/expl_size**2
    
    def update_info(self, event):
        mx=event.x(); my=event.y()
        self.info_label.setText(f"pos: {mx}, {my}")
    
    def set_fluid_dt(self, value):
        self.fluid_dt=value/1000
        self.dt_label.setText(f"Step size: {self.fluid_dt:.3f}min")
        self.fluid_timer.setInterval(int(1000*self.fluid_dt/self.fluid_speed))
    
    def set_fluid_speed(self, value):
        self.fluid_speed=value
        self.speed_label.setText(f"Simulation speed: {value:g}x")
        self.fluid_timer.setInterval(int(1000*self.fluid_dt/self.fluid_speed))
    
    def play_pause(self):
        self.is_running = not self.is_running
    
    def close(self):
        self.closeAllWindows()
    
    def mainloop(self):
        while True:
            pass
        return 0

def main0():
    pg.init()
    pg.font.init()
    fluid_demo=FluidDemo((250, 200))
    return fluid_demo.mainloop()

def main():
    import sys
    sim_app=SimulationApp(sys.argv)
    sim_app.exec()
    #return sim_app.mainloop()

if __name__=="__main__":
    main()
