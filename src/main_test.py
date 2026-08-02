from src.vectors import DesignVector, ParameterVector
from src.main import main
from src.prop.prop_database import load_default_prop_database

dv = DesignVector()
pv = ParameterVector()
prop_database = load_default_prop_database()

print(dv.disp_vars())

total_score, breakdown = main(dv, pv, prop_database=prop_database)

print(f"Total Score: {total_score}")
print(f"Breakdown: {breakdown}")
