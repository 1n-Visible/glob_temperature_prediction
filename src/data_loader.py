from netCDF4 import Dataset
import numpy as np

def load_temperature(filename) -> np.array:
    file=Dataset(filename, "rs") # 's' for unbuffered
    print(file.groups, file.dimensions)
    file.close()
    return None

def main():
    load_temperature("data/data.nc")

if __name__=="__main__":
    main()
