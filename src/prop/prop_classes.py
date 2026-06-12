class propeller:
    def __init__(self, diameter, pitch, mass, idnum):
        self.diameter = diameter #in
        self.pitch = pitch #in
        self.mass = mass #kg
        self.idnum = idnum
class motor:
    def __init__(self, kv, Rm, max_power, I0, max_current, mass):
        self.kv = kv
        self.Rm = Rm
        self.max_power = max_power
        self.I0 = I0
        self.max_current = max_current
        self.mass = mass
class battery:
    def __init__(self, vnom, cells, Rb, Crat, capacity, mass):
        self.vnom = vnom
        self.cells = cells
        self.Rb = Rb
        self.Crat = Crat
        self.capacity = capacity
        self.mass = mass
rho = 1.225 # kg/m^3