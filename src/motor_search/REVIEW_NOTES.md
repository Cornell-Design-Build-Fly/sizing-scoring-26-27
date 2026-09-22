# Review context

The motor-search ranking is not a controlled reproduction of the earlier
continuous sizing optimum. Before selecting hardware, compare these inputs:

- The catalog search includes all 97 retained prop geometries, including
  diameters below the optimizer's 10-inch lower bound. Small props appear in
  the leading results.
- Motor Kv, electrical power limit, resistance, no-load current and mass come
  from the CSV. The earlier sizing path fitted motor resistance, no-load
  current and mass from Kv and power.
- The supplied airframe, fixed 1 kg sensor payloads, explicit tail sizes and
  disabled fuselage/sensor drag are used. Battery capacity is 3.3 Ah rather
  than the supplied optimum's approximately 3.3714 Ah.
- The branch corrects a mechanical-model issue that counted both mission
  propellers on every flight. That changes mass and placement relative to
  previous evaluations.
- The source CSV has no motor maximum-voltage or current ratings. Passing
  the sizing model on an assumed 8S supply does not establish manufacturer
  approval for 8S operation.

The relative contributions of these differences have not been isolated by
a matched-input comparison. The 200–300 Kv inventory in
`results/motors_200_300kv_by_mass.md` is a direct CSV extract, independent of
the propulsion ranking.
