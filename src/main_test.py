from src.vectors import DesignVector, ParameterVector
from src.main import main
from src.prop.continuous_prop_database import load_default_continuous_prop_database


dv = DesignVector()
pv = ParameterVector()
prop_database = load_default_continuous_prop_database()

print(dv.disp_vars())

total_score, breakdown = main(
    dv,
    pv,
    disp_res=False,
    round_payload=True,
    prop_database=prop_database,
)

print(f"Total Score: {total_score}")
print(f"Breakdown: {breakdown}")
