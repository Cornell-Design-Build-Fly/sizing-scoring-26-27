from src.vectors import DesignVector, ParameterVector
from src.main import main
from src.prop.prop_database import load_default_prop_database

dv = DesignVector()
pv = ParameterVector()
prop_database = load_default_prop_database()

print(dv.disp_vars())

<<<<<<< HEAD
total_score, breakdown = main(dv, pv, prop_database=prop_database)
=======
total_score, breakdown = main(dv, pv, False, False)
>>>>>>> d56ba8bdbe889a69d419ff1d501eef4f8e2f9c43

print(f"Total Score: {total_score}")
print(f"Breakdown: {breakdown}")
