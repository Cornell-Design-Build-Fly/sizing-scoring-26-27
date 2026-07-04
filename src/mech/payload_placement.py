"""Mission-2 non-overlapping payload packing with static-margin targeting."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from src.mech.models import (
    MassItem,
    Mission2Config,
    PayloadTypeConfig,
    RelativePayloadRules,
)


class PayloadPlacementError(RuntimeError):
    """Raised when the requested payload cannot fit under the given constraints."""


@dataclass(frozen=True)
class _Candidate:
    payload: PayloadTypeConfig
    position_m: np.ndarray


@dataclass(frozen=True)
class _BeamState:
    selected_mask: int
    selected_indices: tuple[int, ...]
    moment_kg_m: np.ndarray
    compactness_cost: float
    last_local_index: int = -1


def _axis_centers(
    lower: float, upper: float, item_size: float, clearance_m: float
) -> list[float]:
    available = upper - lower
    pitch = item_size + clearance_m
    count = int(np.floor((available + clearance_m + 1e-12) / pitch))
    if count <= 0:
        return []
    occupied = count * item_size + (count - 1) * clearance_m
    offset = 0.5 * (available - occupied)
    return [
        lower + offset + 0.5 * item_size + index * pitch
        for index in range(count)
    ]


def _filter_longitudinal(
    values: Iterable[float],
    item_length: float,
    reference_x: float,
    payload: PayloadTypeConfig,
) -> list[float]:
    result: list[float] = []
    half = 0.5 * item_length
    for value in values:
        forward = value + half <= reference_x + 1e-12
        aft = value - half >= reference_x - 1e-12
        if (payload.rules.allow_forward and forward) or (payload.rules.allow_aft and aft):
            result.append(value)
    return result


def _filter_vertical(
    values: Iterable[float],
    item_height: float,
    reference_z: float,
    payload: PayloadTypeConfig,
) -> list[float]:
    half = 0.5 * item_height
    above_values: list[float] = []
    below_values: list[float] = []
    for value in values:
        if value - half >= reference_z - 1e-12:
            above_values.append(value)
        if value + half <= reference_z + 1e-12:
            below_values.append(value)

    if not payload.rules.allow_stacking:
        # One layer per permitted side, selected nearest the reference plane.
        above_values = [min(above_values)] if above_values else []
        below_values = [max(below_values)] if below_values else []

    result: list[float] = []
    if payload.rules.allow_above:
        result.extend(above_values)
    if payload.rules.allow_below:
        result.extend(below_values)
    return sorted(set(result))


def _generate_candidates(
    payload: PayloadTypeConfig,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    reference_x: float,
    reference_z: float,
    clearance_m: float,
) -> list[_Candidate]:
    lx, ly, lz = payload.dimensions_m
    xs = _axis_centers(*x_bounds, lx, clearance_m)
    ys = _axis_centers(*y_bounds, ly, clearance_m)
    zs = _axis_centers(*z_bounds, lz, clearance_m)

    xs = _filter_longitudinal(xs, lx, reference_x, payload)
    zs = _filter_vertical(zs, lz, reference_z, payload)

    return [
        _Candidate(payload=payload, position_m=np.array([x, y, z], dtype=float))
        for x in xs
        for y in ys
        for z in zs
    ]


def _boxes_conflict(a: _Candidate, b: _Candidate, clearance_m: float) -> bool:
    # Boxes may touch only after the requested clearance is included.
    half_a = 0.5 * np.asarray(a.payload.dimensions_m)
    half_b = 0.5 * np.asarray(b.payload.dimensions_m)
    required_separation = half_a + half_b + clearance_m
    return bool(
        np.all(np.abs(a.position_m - b.position_m) < required_separation - 1e-12)
    )


def _trim_candidates(
    candidates: list[_Candidate],
    maximum: int,
    desired_payload_cg_x_m: float,
    center_y_m: float,
    reference_z_m: float,
) -> list[_Candidate]:
    if len(candidates) <= maximum:
        return candidates

    ranked = sorted(
        enumerate(candidates),
        key=lambda pair: (
            abs(pair[1].position_m[0] - desired_payload_cg_x_m),
            abs(pair[1].position_m[1] - center_y_m)
            + abs(pair[1].position_m[2] - reference_z_m),
            pair[0],
        ),
    )
    return [candidate for _, candidate in ranked[:maximum]]


def _relative_rule_conflict(
    a: _Candidate,
    b: _Candidate,
    rules: RelativePayloadRules,
    clearance_m: float,
) -> bool:
    labels = {a.payload.label.lower(), b.payload.label.lower()}
    if labels != {"duck", "puck"}:
        return False

    puck = a if a.payload.label.lower() == "puck" else b
    duck = b if puck is a else a
    puck_half = 0.5 * np.asarray(puck.payload.dimensions_m)
    duck_half = 0.5 * np.asarray(duck.payload.dimensions_m)
    puck_lower = puck.position_m - puck_half
    puck_upper = puck.position_m + puck_half
    duck_lower = duck.position_m - duck_half
    duck_upper = duck.position_m + duck_half

    if (
        rules.pucks_forward_of_ducks
        and puck_upper[0] + clearance_m > duck_lower[0] + 1e-12
    ):
        return True
    if (
        rules.pucks_aft_of_ducks
        and puck_lower[0] - clearance_m < duck_upper[0] - 1e-12
    ):
        return True
    if (
        rules.pucks_above_ducks
        and puck_lower[2] - clearance_m < duck_upper[2] - 1e-12
    ):
        return True
    if (
        rules.pucks_below_ducks
        and puck_upper[2] + clearance_m > duck_lower[2] + 1e-12
    ):
        return True
    return False


def _conflict_masks(
    candidates: list[_Candidate],
    clearance_m: float,
    relative_rules: RelativePayloadRules,
) -> list[int]:
    masks = [0 for _ in candidates]
    for first, second in combinations(range(len(candidates)), 2):
        if _boxes_conflict(
            candidates[first], candidates[second], clearance_m
        ) or _relative_rule_conflict(
            candidates[first], candidates[second], relative_rules, clearance_m
        ):
            masks[first] |= 1 << second
            masks[second] |= 1 << first
    return masks


def _remaining_moment_interval(
    *,
    selected_mask: int,
    group_start_index: int,
    group_progress: int,
    groups: list[tuple[PayloadTypeConfig, int, list[int]]],
    conflict_masks: list[int],
    candidates: list[_Candidate],
) -> tuple[np.ndarray, np.ndarray] | None:
    minimum = np.zeros(3)
    maximum = np.zeros(3)
    for group_index in range(group_start_index, len(groups)):
        payload, total_count, indices = groups[group_index]
        remaining_count = total_count - (group_progress if group_index == group_start_index else 0)
        if remaining_count <= 0:
            continue
        available = [
            index
            for index in indices
            if not (selected_mask & (1 << index))
            and not (conflict_masks[index] & selected_mask)
        ]
        if len(available) < remaining_count:
            return None
        positions = np.array([candidates[index].position_m for index in available])
        mass = payload.mass_kg
        for axis in range(3):
            axis_values = np.sort(positions[:, axis])
            minimum[axis] += mass * np.sum(axis_values[:remaining_count])
            maximum[axis] += mass * np.sum(axis_values[-remaining_count:])
    return minimum, maximum


def _beam_score(
    state: _BeamState,
    desired_moment_kg_m: np.ndarray,
    remaining_interval: tuple[np.ndarray, np.ndarray],
    config: Mission2Config,
) -> float:
    minimum, maximum = remaining_interval
    final_minimum = state.moment_kg_m + minimum
    final_maximum = state.moment_kg_m + maximum
    lower_bound_error = np.maximum(
        final_minimum - desired_moment_kg_m,
        desired_moment_kg_m - final_maximum,
    )
    lower_bound_error = np.maximum(lower_bound_error, 0.0)

    midpoint = 0.5 * (final_minimum + final_maximum)
    midpoint_error = np.abs(midpoint - desired_moment_kg_m)

    # Longitudinal balance is primary. Lateral and vertical moments break ties
    # toward centered, symmetric arrangements.
    return float(
        1000.0 * lower_bound_error[0]
        + 10.0 * lower_bound_error[1]
        + lower_bound_error[2]
        + 0.02 * midpoint_error[0]
        + 0.002 * midpoint_error[1]
        + 0.0002 * midpoint_error[2]
        + config.compactness_weight * state.compactness_cost
    )



def _greedy_orders(groups: list[tuple[PayloadTypeConfig, int, list[int]]]) -> list[list[int]]:
    """Return deterministic payload-type sequences for multi-start packing."""
    if len(groups) != 2:
        # The current competition model has ducks and pucks. Keep a generic
        # fallback in case another type is added later.
        sequence: list[int] = []
        for group_index, (_, count, _) in enumerate(groups):
            sequence.extend([group_index] * count)
        return [sequence]

    first, second = groups
    count_first, count_second = first[1], second[1]
    orders: list[list[int]] = [
        [0] * count_first + [1] * count_second,
        [1] * count_second + [0] * count_first,
    ]

    # Heavier remaining group first.
    weighted: list[int] = []
    remaining = [count_first, count_second]
    while sum(remaining):
        options = [index for index, count in enumerate(remaining) if count > 0]
        chosen = max(
            options,
            key=lambda index: remaining[index] * groups[index][0].mass_kg,
        )
        weighted.append(chosen)
        remaining[chosen] -= 1
    orders.append(weighted)

    # Simple alternating order, starting with the second (normally heavier) type.
    alternating: list[int] = []
    for item_index in range(max(count_first, count_second)):
        if item_index < count_second:
            alternating.append(1)
        if item_index < count_first:
            alternating.append(0)
    orders.append(alternating)

    unique: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for order in orders:
        key = tuple(order)
        if key not in seen:
            seen.add(key)
            unique.append(order)
    return unique


def _select_with_greedy_multistart(
    *,
    candidates: list[_Candidate],
    groups: list[tuple[PayloadTypeConfig, int, list[int]]],
    conflict_masks: list[int],
    desired_moment_kg_m: np.ndarray,
    config: Mission2Config,
) -> list[int]:
    """Fast deterministic packing with several order and layer preferences."""

    best: tuple[float, tuple[int, ...]] | None = None
    # Positive means prefer above, negative means prefer below. These starts
    # encourage the two item types to occupy different vertical layers when that
    # improves packing, while still allowing a neutral unconstrained start.
    vertical_preferences = [
        (0.0, 0.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (1.0, 1.0),
        (-1.0, -1.0),
    ]

    for order in _greedy_orders(groups):
        for preference_pair in vertical_preferences:
            selected: list[int] = []
            selected_mask = 0
            selected_by_group = [set() for _ in groups]
            remaining = [group[1] for group in groups]
            payload_moment = np.zeros(3)
            feasible_start = True

            for group_index in order:
                remaining[group_index] -= 1
                payload, _, group_indices = groups[group_index]
                choices: list[tuple[float, int, np.ndarray, int]] = []

                for candidate_index in group_indices:
                    if candidate_index in selected_by_group[group_index]:
                        continue
                    if conflict_masks[candidate_index] & selected_mask:
                        continue

                    new_mask = selected_mask | (1 << candidate_index)
                    capacity_ok = True
                    for future_group_index, (_, _, future_indices) in enumerate(groups):
                        available_count = sum(
                            1
                            for future_index in future_indices
                            if future_index not in selected_by_group[future_group_index]
                            and future_index != candidate_index
                            and not (conflict_masks[future_index] & new_mask)
                        )
                        if available_count < remaining[future_group_index]:
                            capacity_ok = False
                            break
                    if not capacity_ok:
                        continue

                    candidate = candidates[candidate_index]
                    new_moment = (
                        payload_moment + payload.mass_kg * candidate.position_m
                    )

                    # Tight lower bound on the longitudinal moment error after
                    # filling all remaining identical-item slots.
                    min_remaining_x_moment = 0.0
                    max_remaining_x_moment = 0.0
                    for future_group_index, (future_payload, _, future_indices) in enumerate(groups):
                        needed = remaining[future_group_index]
                        if needed == 0:
                            continue
                        available_x = sorted(
                            candidates[future_index].position_m[0]
                            for future_index in future_indices
                            if future_index not in selected_by_group[future_group_index]
                            and future_index != candidate_index
                            and not (conflict_masks[future_index] & new_mask)
                        )
                        if len(available_x) < needed:
                            capacity_ok = False
                            break
                        min_remaining_x_moment += future_payload.mass_kg * sum(
                            available_x[:needed]
                        )
                        max_remaining_x_moment += future_payload.mass_kg * sum(
                            available_x[-needed:]
                        )
                    if not capacity_ok:
                        continue

                    lower_bound_x_error = max(
                        new_moment[0]
                        + min_remaining_x_moment
                        - desired_moment_kg_m[0],
                        desired_moment_kg_m[0]
                        - (new_moment[0] + max_remaining_x_moment),
                        0.0,
                    )

                    vertical_preference = preference_pair[
                        min(group_index, len(preference_pair) - 1)
                    ]
                    wrong_vertical_side = (
                        0.0
                        if vertical_preference == 0.0
                        else max(
                            0.0,
                            -vertical_preference * candidate.position_m[2],
                        )
                    )
                    future_blocked = sum(
                        1
                        for future_group_index, (_, _, future_indices) in enumerate(groups)
                        for future_index in future_indices
                        if remaining[future_group_index] > 0
                        and (conflict_masks[candidate_index] & (1 << future_index))
                    )

                    score = (
                        1000.0 * lower_bound_x_error
                        + 0.10 * abs(new_moment[1] - desired_moment_kg_m[1])
                        + 0.01 * abs(new_moment[2] - desired_moment_kg_m[2])
                        + 0.01 * wrong_vertical_side
                        + 1e-5 * future_blocked
                        + config.compactness_weight
                        * (
                            abs(candidate.position_m[1] - config.compartment_center_y_m)
                            + abs(candidate.position_m[2] - config.relative_reference_z_m)
                        )
                    )
                    choices.append((score, candidate_index, new_moment, new_mask))

                if not choices:
                    feasible_start = False
                    break

                choices.sort(key=lambda choice: (choice[0], choice[1]))
                _, selected_index, payload_moment, selected_mask = choices[0]
                selected.append(selected_index)
                selected_by_group[group_index].add(selected_index)

            if not feasible_start:
                continue

            moment_error = np.abs(payload_moment - desired_moment_kg_m)
            final_score = float(
                moment_error[0] + 0.10 * moment_error[1] + 0.01 * moment_error[2]
            )
            selected_key = tuple(sorted(selected))
            candidate_solution = (final_score, selected_key)
            if best is None or candidate_solution < best:
                best = candidate_solution

    if best is None:
        raise PayloadPlacementError(
            "Greedy multi-start packing could not find a feasible non-overlapping arrangement."
        )
    return list(best[1])

def _select_with_beam_search(
    *,
    candidates: list[_Candidate],
    groups: list[tuple[PayloadTypeConfig, int, list[int]]],
    conflict_masks: list[int],
    desired_moment_kg_m: np.ndarray,
    config: Mission2Config,
) -> list[int]:
    # Put the most constrained/largest payload group first.
    groups = sorted(
        groups,
        key=lambda group: (
            len(group[2]) / max(group[1], 1),
            -np.prod(group[0].dimensions_m),
        ),
    )

    states = [
        _BeamState(
            selected_mask=0,
            selected_indices=(),
            moment_kg_m=np.zeros(3),
            compactness_cost=0.0,
        )
    ]

    for group_index, (payload, count, indices) in enumerate(groups):
        if count == 0:
            continue
        states = [
            _BeamState(
                selected_mask=state.selected_mask,
                selected_indices=state.selected_indices,
                moment_kg_m=state.moment_kg_m,
                compactness_cost=state.compactness_cost,
                last_local_index=-1,
            )
            for state in states
        ]

        for placed_in_group in range(count):
            expanded: list[tuple[float, _BeamState]] = []
            remaining_after_choice = count - placed_in_group - 1

            for state in states:
                feasible_local: list[tuple[float, int, int]] = []
                for local_index in range(state.last_local_index + 1, len(indices)):
                    candidate_index = indices[local_index]
                    if state.selected_mask & (1 << candidate_index):
                        continue
                    if conflict_masks[candidate_index] & state.selected_mask:
                        continue

                    # Enough later candidates from this identical group must
                    # remain to complete a combination without permutations.
                    later_available = 0
                    for later_local in range(local_index + 1, len(indices)):
                        later_index = indices[later_local]
                        prospective_mask = state.selected_mask | (1 << candidate_index)
                        if not (conflict_masks[later_index] & prospective_mask):
                            later_available += 1
                    if later_available < remaining_after_choice:
                        continue

                    candidate = candidates[candidate_index]
                    local_moment = state.moment_kg_m + payload.mass_kg * candidate.position_m
                    x_need = desired_moment_kg_m[0] - local_moment[0]
                    local_cost = abs(x_need)
                    local_cost += 0.05 * abs(local_moment[1])
                    local_cost += 0.005 * abs(local_moment[2])
                    # Prefer candidates that block fewer future-type locations.
                    future_blocked = 0
                    for _, _, future_indices in groups[group_index + 1 :]:
                        future_blocked += sum(
                            bool(conflict_masks[candidate_index] & (1 << future_index))
                            for future_index in future_indices
                        )
                    local_cost += 1e-5 * future_blocked
                    feasible_local.append((local_cost, local_index, candidate_index))

                feasible_local.sort(key=lambda entry: (entry[0], entry[1]))
                for _, local_index, candidate_index in feasible_local[
                    : config.branch_limit_per_state
                ]:
                    candidate = candidates[candidate_index]
                    new_mask = state.selected_mask | (1 << candidate_index)
                    new_state = _BeamState(
                        selected_mask=new_mask,
                        selected_indices=state.selected_indices + (candidate_index,),
                        moment_kg_m=state.moment_kg_m
                        + payload.mass_kg * candidate.position_m,
                        compactness_cost=state.compactness_cost
                        + abs(candidate.position_m[1] - config.compartment_center_y_m)
                        + abs(candidate.position_m[2] - config.relative_reference_z_m),
                        last_local_index=local_index,
                    )
                    interval = _remaining_moment_interval(
                        selected_mask=new_mask,
                        group_start_index=group_index,
                        group_progress=placed_in_group + 1,
                        groups=groups,
                        conflict_masks=conflict_masks,
                        candidates=candidates,
                    )
                    if interval is None:
                        continue
                    expanded.append(
                        (
                            _beam_score(
                                new_state, desired_moment_kg_m, interval, config
                            ),
                            new_state,
                        )
                    )

            if not expanded:
                raise PayloadPlacementError(
                    "Beam search could not find a feasible non-overlapping payload arrangement."
                )

            expanded.sort(key=lambda entry: (entry[0], entry[1].selected_indices))
            unique: dict[int, _BeamState] = {}
            for _, state in expanded:
                unique.setdefault(state.selected_mask, state)
                if len(unique) >= config.beam_width:
                    break
            states = list(unique.values())

        # Reset combination-order bookkeeping before the next payload type.
        states = [
            _BeamState(
                selected_mask=state.selected_mask,
                selected_indices=state.selected_indices,
                moment_kg_m=state.moment_kg_m,
                compactness_cost=state.compactness_cost,
            )
            for state in states
        ]

    if not states:
        raise PayloadPlacementError("No feasible payload arrangement was found.")

    def final_score(state: _BeamState) -> tuple[float, float, tuple[int, ...]]:
        error = np.abs(state.moment_kg_m - desired_moment_kg_m)
        return (
            float(error[0] + 0.10 * error[1] + 0.01 * error[2]),
            state.compactness_cost,
            state.selected_indices,
        )

    best = min(states, key=final_score)
    return list(best.selected_indices)


def _select_with_milp(
    *,
    candidates: list[_Candidate],
    groups: list[tuple[PayloadTypeConfig, int, list[int]]],
    conflict_masks: list[int],
    desired_moment_kg_m: np.ndarray,
    config: Mission2Config,
) -> list[int]:
    binary_count = len(candidates)
    # Three absolute-moment-error variables: x, y, and z.
    error_start = binary_count
    variable_count = binary_count + 3

    objective = np.zeros(variable_count)
    objective[error_start : error_start + 3] = [1.0, 0.10, 0.01]
    for index, candidate in enumerate(candidates):
        objective[index] = config.compactness_weight * (
            abs(candidate.position_m[1] - config.compartment_center_y_m)
            + abs(candidate.position_m[2] - config.relative_reference_z_m)
        )

    rows: list[tuple[dict[int, float], float, float]] = []
    for _, count, indices in groups:
        if count:
            rows.append(({index: 1.0 for index in indices}, count, count))

    for first, second in combinations(range(binary_count), 2):
        if conflict_masks[first] & (1 << second):
            rows.append(({first: 1.0, second: 1.0}, -np.inf, 1.0))

    for axis in range(3):
        coefficients = {
            index: candidate.payload.mass_kg * candidate.position_m[axis]
            for index, candidate in enumerate(candidates)
        }
        positive = dict(coefficients)
        positive[error_start + axis] = -1.0
        rows.append((positive, -np.inf, desired_moment_kg_m[axis]))
        negative = {index: -value for index, value in coefficients.items()}
        negative[error_start + axis] = -1.0
        rows.append((negative, -np.inf, -desired_moment_kg_m[axis]))

    matrix = lil_matrix((len(rows), variable_count), dtype=float)
    lower = np.empty(len(rows))
    upper = np.empty(len(rows))
    for row_index, (coefficients, row_lower, row_upper) in enumerate(rows):
        for column_index, coefficient in coefficients.items():
            matrix[row_index, column_index] = coefficient
        lower[row_index] = row_lower
        upper[row_index] = row_upper

    result = milp(
        c=objective,
        integrality=np.concatenate(
            [np.ones(binary_count, dtype=int), np.zeros(3, dtype=int)]
        ),
        bounds=Bounds(
            lb=np.concatenate([np.zeros(binary_count), np.zeros(3)]),
            ub=np.concatenate([np.ones(binary_count), np.full(3, np.inf)]),
        ),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"disp": False, "time_limit": config.milp_time_limit_s},
    )
    if not result.success or result.x is None:
        raise PayloadPlacementError(
            "The Mission-2 MILP did not find a feasible arrangement within the "
            f"configured limit (solver status: {result.message})."
        )
    return [index for index in range(binary_count) if result.x[index] >= 0.5]


def place_mission2_payload(
    *,
    duck_count: int,
    puck_count: int,
    base_items: Iterable[MassItem],
    target_cg_x_m: float,
    config: Mission2Config,
    x_bounds_m: tuple[float, float],
    reference_x_m: float,
) -> tuple[MassItem, ...]:
    """Place all M2 items without overlap while targeting static margin."""

    if duck_count < 0 or puck_count < 0:
        raise ValueError("Payload counts cannot be negative.")
    if duck_count == 0 and puck_count == 0:
        return ()

    x_min, x_max = x_bounds_m
    if not x_min < x_max:
        raise ValueError("Mission-2 compartment x bounds must be increasing.")

    y_half = 0.5 * config.maximum_width_m
    z_half = 0.5 * config.maximum_height_m
    y_bounds = (
        config.compartment_center_y_m - y_half,
        config.compartment_center_y_m + y_half,
    )
    z_bounds = (
        config.compartment_center_z_m - z_half,
        config.compartment_center_z_m + z_half,
    )

    base_items = tuple(base_items)
    base_mass = sum(item.mass_kg for item in base_items)
    base_moment = sum(
        (item.mass_kg * item.position_m for item in base_items), start=np.zeros(3)
    )
    payload_mass = duck_count * config.duck.mass_kg + puck_count * config.puck.mass_kg
    desired_total_moment = np.array(
        [
            target_cg_x_m * (base_mass + payload_mass),
            0.0,
            base_moment[2],
        ]
    )
    desired_payload_moment = desired_total_moment - base_moment
    desired_payload_cg_x = desired_payload_moment[0] / payload_mass

    candidates: list[_Candidate] = []
    groups: list[tuple[PayloadTypeConfig, int, list[int]]] = []
    for payload, count in ((config.duck, duck_count), (config.puck, puck_count)):
        generated = _generate_candidates(
            payload,
            x_bounds=(x_min, x_max),
            y_bounds=y_bounds,
            z_bounds=z_bounds,
            reference_x=reference_x_m,
            reference_z=config.relative_reference_z_m,
            clearance_m=config.clearance_m,
        )
        generated = _trim_candidates(
            generated,
            maximum=config.max_candidates_per_type,
            desired_payload_cg_x_m=desired_payload_cg_x,
            center_y_m=config.compartment_center_y_m,
            reference_z_m=config.relative_reference_z_m,
        )
        if len(generated) < count:
            raise PayloadPlacementError(
                f"Only {len(generated)} candidate locations are available for {count} "
                f"{payload.label.lower()} items. Increase compartment dimensions or relax "
                "the placement/stacking rules."
            )
        start = len(candidates)
        candidates.extend(generated)
        groups.append((payload, count, list(range(start, len(candidates)))))

    conflicts = _conflict_masks(
        candidates, config.clearance_m, config.relative_payload_rules
    )

    if config.solver == "greedy":
        selected_indices = _select_with_greedy_multistart(
            candidates=candidates,
            groups=groups,
            conflict_masks=conflicts,
            desired_moment_kg_m=desired_payload_moment,
            config=config,
        )
    elif config.solver == "beam":
        selected_indices = _select_with_beam_search(
            candidates=candidates,
            groups=groups,
            conflict_masks=conflicts,
            desired_moment_kg_m=desired_payload_moment,
            config=config,
        )
    elif config.solver == "milp":
        selected_indices = _select_with_milp(
            candidates=candidates,
            groups=groups,
            conflict_masks=conflicts,
            desired_moment_kg_m=desired_payload_moment,
            config=config,
        )
    else:
        try:
            selected_indices = _select_with_greedy_multistart(
                candidates=candidates,
                groups=groups,
                conflict_masks=conflicts,
                desired_moment_kg_m=desired_payload_moment,
                config=config,
            )
        except PayloadPlacementError:
            try:
                selected_indices = _select_with_beam_search(
                    candidates=candidates,
                    groups=groups,
                    conflict_masks=conflicts,
                    desired_moment_kg_m=desired_payload_moment,
                    config=config,
                )
            except PayloadPlacementError:
                selected_indices = _select_with_milp(
                    candidates=candidates,
                    groups=groups,
                    conflict_masks=conflicts,
                    desired_moment_kg_m=desired_payload_moment,
                    config=config,
                )

    selected = [candidates[index] for index in selected_indices]
    expected = duck_count + puck_count
    if len(selected) != expected:
        raise PayloadPlacementError(
            f"Placement selected {len(selected)} payloads, but {expected} were required."
        )

    placed_items: list[MassItem] = []
    counters: dict[str, int] = {}
    for candidate in sorted(
        selected,
        key=lambda item: (
            item.payload.label,
            item.position_m[0],
            item.position_m[1],
            item.position_m[2],
        ),
    ):
        counters[candidate.payload.label] = counters.get(candidate.payload.label, 0) + 1
        placed_items.append(
            MassItem(
                name=f"{candidate.payload.label} {counters[candidate.payload.label]:02d}",
                mass_kg=candidate.payload.mass_kg,
                position_m=candidate.position_m,
                dimensions_m=candidate.payload.dimensions_m,
                missions=frozenset({"M2"}),
                category="mission_2_payload",
            )
        )
    return tuple(placed_items)
