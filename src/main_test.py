from src.vectors import DesignVector, ParameterVector
from src.main import main

dv = DesignVector()
pv = ParameterVector()

print(dv.disp_vars())

total_score, breakdown = main(dv, pv)
print(f"Total Score: {total_score}")
print(f"Breakdown: {breakdown}")
