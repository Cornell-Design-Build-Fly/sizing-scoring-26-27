#!/bin/zsh
cd /Users/ishanroy/Documents/GitHub/sizing-scoring-26-27
# geometry_test.py and vector_test.py match pytest's *_test.py pattern but are
# demo scripts that open a blocking three-view window at import; excluded.
for m in mech_test test_flap_model opt_score_test test_aero_endurance test_continuous_lap_scoring test_niching_islands test_propulsion_requirements test_spiral_criterion test_stratified_optimizer_init test_top_candidate_archive test_tow_envelope test_turn_energy_model topline_payload_archive_test; do
  out=$(PYTHONPATH=. venv/bin/python -m pytest "src/testing/$m.py" -q -p no:cacheprovider 2>&1 | grep -E "passed|failed|error|no tests ran" | tail -1)
  printf "%-36s %s\n" "$m" "$out"
done
echo "ALL_MODULES_DONE"
