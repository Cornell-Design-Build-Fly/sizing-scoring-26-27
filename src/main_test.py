from time import perf_counter

from src.vectors import DesignVector, ParameterVector
from src.main import main

dv = DesignVector()
pv = ParameterVector()

print(dv.disp_vars())

module_timings: dict[str, float] = {}
total_start = perf_counter()
total_score, breakdown = main(dv, pv, module_timings=module_timings)
total_elapsed = perf_counter() - total_start

print(f"Total Score: {total_score}")
print(f"Breakdown: {breakdown}")
print("\nModule timing summary:")
for module_name, elapsed_seconds in module_timings.items():
    print(f"  {module_name.title():<13} {elapsed_seconds:>9.3f} s")
print(f"  {'Total':<13} {total_elapsed:>9.3f} s")
