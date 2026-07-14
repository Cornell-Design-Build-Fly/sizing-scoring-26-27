"""
Aero scoring test.

Tests the full chain:
    DesignVector  →  stability_analysis()  →  aero_score()

Four sub-tests:

  1. Stable baseline   — default DesignVector + stub mass; verifies can_fly=True.
  2. Unstable design   — StabilityResult with Cma>0, SM<0; verifies penalty>0.
  2b. Spiral criterion — statically stable but spiral dt<4 s; verifies can_fly=False.
  3. DF1 real params   — Duck Force One with XFLR5-extracted geometry/inertia and
                         report CG/weight/speed for M2 and M3.

Run from repo root:
    python3 src/testing/aero_score_test.py

XFLR5 geometry notes (Mission 2 Iteration 2.xml, Mission 3 Iteration 2.xml):
  Wing:       chord=0.307 m, half-span=0.591 m, airfoil=FX 60-126
  Elevator:   chord=0.152 m, half-span=0.219 m, LE at x=0.830 (M2) / 0.845 (M3)
  Fin:        chord≈0.148 m, span≈0.132 m,     LE at same x as elevator
  tail_arm:   wing_QC→HT_QC = 0.791 m (M2),  0.806 m (M3)

  NOTE: V_V=0.036, AR_V=0.89 in vectors.py produce VT dimensions matching the
        actual DF1 fin (5.18 in × 5.83 in vs XFLR5's 5.24 in × 5.74 in).
        Stability solver uses VLM (not AeroBuildup) to capture wing-tail downwash.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import aerosandbox as asb
import numpy as np

from src.vectors import ASBDesignVector, DesignVector
from src.aero.custom_classes import CruiseCondition, StabilityResult
from src.aero.stability_analysis import stability_analysis
from src.aero.vlm import AirplaneAnalysisResult
from src.aero.aero_score import aero_score, AeroScore, SPIRAL_RATE_MAX, _compute_lap_time

# ── Shared test parameters ─────────────────────────────────────────────────
CRUISE_SPEED = 14.0  # m/s

# ── XFLR5 mass data ────────────────────────────────────────────────────────
# Coordinates: origin=wing LE, +x aft, +z up [meters]
# Tail surfaces: volume mass approximated as point mass at geometric centroid.
# Wing and elevator span distribution is corrected analytically in _xflr5_inertia().

_BODY_BASE = [
    (0.163, [ 0.415,  0.,   0.   ]),  # tail spar
    (0.337, [ 0.089,  0.,   0.   ]),  # wing integration + spar
    (0.024, [-0.007,  0.,  -0.070]),  # receiver + satellite
    (0.045, [-0.007,  0.,  -0.070]),  # servo battery
    (0.390, [-0.222,  0.,  -0.070]),  # motor + prop
    (0.021, [-0.007,  0.,  -0.070]),  # gyroscope
    (0.690, [-0.062,  0.,  -0.070]),  # main battery
    (0.152, [-0.161,  0.,  -0.070]),  # nosecone
    (0.125, [ 0.120,  0.,  -0.210]),  # main gear
    (0.105, [-0.107,  0.,  -0.210]),  # nose gear
    (0.385, [ 0.127,  0.,  -0.070]),  # fuselage
    (0.398, [ 0.138,  0.,   0.   ]),  # wing surface mass (at x-centroid)
]

# M2: elevator LE=0.830, chord=0.152 → centroid at x=0.906
#     fin LE=0.830, chord=0.146, span=0.133 → centroid at (0.903, 0, 0.067)
_M2_MASSES = _BODY_BASE + [
    (0.116, [ 0.191,  0.,  -0.070]),  # duck payload
    (0.190, [ 0.084,  0.,  -0.070]),  # puck payload
    (0.089, [ 0.906,  0.,   0.   ]),  # elevator volume mass
    (0.049, [ 0.903,  0.,   0.067]),  # fin volume mass
]

# M3: elevator LE=0.845, chord=0.152 → centroid at x=0.921
#     fin LE=0.845, chord=0.150, span=0.130 → centroid at (0.920, 0, 0.065)
_M3_MASSES = _BODY_BASE + [
    (0.136, [ 0.518,  0.,   0.   ]),  # banner drop mechanism
    (0.300, [ 0.518,  0.,  -0.070]),  # banner
    (0.089, [ 0.921,  0.,   0.   ]),  # elevator volume mass
    (0.049, [ 0.920,  0.,   0.065]),  # fin volume mass
]

# Wing and elevator geometry for span correction
_WING_MASS_KG     = 0.398
_WING_HALF_SPAN_M = 0.591
_ELEV_MASS_KG     = 0.089
_ELEV_HALF_SPAN_M = 0.219


def _xflr5_inertia(masses, cg_override=None):
    """
    Compute (total_mass_kg, cg_m, inertia_3x3_kgm2) from an XFLR5 mass list.

    All XFLR5 mass entries have y_i = 0 (symmetric aircraft), so dy=0 in the
    point-mass sums.  Ixx and Izz are corrected for the actual spanwise
    distribution of the wing and elevator via I_span = m*(b/2)^2/3.

    cg_override: if given, compute inertia about this point instead of the
                 mass-list centroid (useful for M3 where XFLR5 CG ≠ report CG).
    """
    total = sum(m for m, _ in masses)
    mass_cg = np.array([
        sum(m * r[0] for m, r in masses) / total,
        0.,
        sum(m * r[2] for m, r in masses) / total,
    ])
    cg = np.array(cg_override) if cg_override is not None else mass_cg

    Ixx = Iyy = Izz = 0.
    for m, r in masses:
        dx = r[0] - cg[0]
        dz = r[2] - cg[2]
        Ixx += m * dz * dz
        Iyy += m * (dx * dx + dz * dz)
        Izz += m * dx * dx

    # Span correction: uniform distribution over ±b/2 contributes m*(b/2)^2/3
    span_corr = (_WING_MASS_KG * _WING_HALF_SPAN_M**2 / 3 +
                 _ELEV_MASS_KG * _ELEV_HALF_SPAN_M**2 / 3)
    Ixx += span_corr
    Izz += span_corr

    return total, mass_cg, np.diag([Ixx, Iyy, Izz])


# ── Stub aero result (not used internally by stability_analysis) ───────────
_AERO_STUB = AirplaneAnalysisResult(
    CL=0.3, CD=0.04, CY=0., Cl=0., Cm=-0.1, Cn=0.,
    L=30., D=4., Y=0., l_b=0., m_b=-0.3, n_b=0.,
    runtime_seconds=0., converged=True,
)


_STABLE_TEST_SPEED = 25.0   # m/s — fast enough for the spiral mode to behave realistically.
                             # The old stub used 14 m/s with Iyy=0.02 kg·m², giving a spiral
                             # eigenvalue of +33 s⁻¹ (unphysical). With XFLR5 inertia and a
                             # speed closer to M2 (27.7 m/s), the spiral is physical and passes.


def _build_stability_result():
    """
    Build a plane from the default DesignVector, run stability_analysis,
    and return (StabilityResult, cruise_speed).

    Uses XFLR5 M2 inertia (Ixx=0.057, Iyy=0.188, Izz=0.226 kg·m²) and a
    physically realistic CG.  The tiny stub inertia (Iyy=0.02) that was here
    before made the spiral eigenvalue +33 s⁻¹ — unphysical and now correctly
    flagged as a can_fly failure now that the spiral criterion is enforced.
    """
    dv     = DesignVector()
    asb_dv = ASBDesignVector.from_design_vector(dv)

    print("=== Aero Score Test ===")
    print(f"Wing span:  {dv.wing_span:.3f} m")
    print(f"Wing chord: {dv.wing_chord:.3f} m")
    print(f"Tail arm:   {dv.tail_arm:.3f} m  (default; XFLR5 actual = 0.791 m)")

    # CG: close to M2 XFLR5 position (x=0.065 m from wing LE, z=-0.056 m)
    cg     = [0.065, 0.0, -0.056]
    weight = 3.0 * 9.806   # ~3 kg total (realistic for DF1 range)

    # Inertia from XFLR5 M2 mass breakdown (same as DF1 test, minus payload scaling).
    # These realistic values prevent the spiral from collapsing to an unphysical rate.
    inertia_matrix = np.array([
        [0.057, 0.,    0.   ],  # Ixx (roll)  — wing span dominated
        [0.,    0.188, 0.   ],  # Iyy (pitch) — motor + tail arm dominated
        [0.,    0.,    0.226],  # Izz (yaw)
    ])

    op_point    = asb.OperatingPoint(velocity=_STABLE_TEST_SPEED, alpha=1.5)
    cruise_cond = CruiseCondition(operating_point=op_point, throttle=0.6)

    print("\nRunning stability_analysis()...")
    stab = stability_analysis(dv, cruise_cond, _AERO_STUB, cg, inertia_matrix, weight)
    return stab, _STABLE_TEST_SPEED


# ──────────────────────────────────────────────────────────────────────────
# Test 1: stable baseline
# ──────────────────────────────────────────────────────────────────────────

def test_stable_design() -> AeroScore:
    """Default DesignVector → can_fly=True, penalty=0."""
    stab, cruise_speed = _build_stability_result()

    print("\n── Static stability ──────────────────────────────────────")
    print(f"  Cma:           {stab.Cma:.4f}  ({'OK' if stab.Cma < 0 else 'FAIL'})")
    print(f"  Cnb:           {stab.Cnb:.4f}  ({'OK' if stab.Cnb > 0 else 'FAIL'})")
    print(f"  Static margin: {stab.static_margin:.4f}  ({'OK' if stab.static_margin > 0 else 'FAIL'})")

    spiral_rate = stab.spiral_eigenvalue.real
    spiral_dt   = np.log(2) / spiral_rate if spiral_rate > 0 else float('inf')
    dt_str = f"{spiral_dt:.1f} s" if np.isfinite(spiral_dt) else "∞"
    print(f"  Spiral λ:      {spiral_rate:+.4f} s⁻¹  (dt={dt_str}, limit=4.0 s, "
          f"{'OK' if spiral_rate <= SPIRAL_RATE_MAX else 'FAIL'})")

    print("\n── Dynamic modes ─────────────────────────────────────────")
    modes = [
        ("Phugoid",         stab.phugoid_eigenvalue,        stab.phugoid_damping_ratio),
        ("Short period",    stab.short_period_eigenvalue,   stab.short_period_damping_ratio),
        ("Roll subsidence", stab.roll_subsidence_eigenvalue,stab.roll_subsidence_damping_ratio),
        ("Dutch roll",      stab.dutch_roll_eigenvalue,     stab.dutch_roll_damping_ratio),
        ("Spiral",          stab.spiral_eigenvalue,         stab.spiral_damping_ratio),
    ]
    print(f"  {'Mode':<20} {'Re(λ)':>10} {'Im(λ)':>10} {'ζ':>8}  Status")
    print("  " + "-" * 58)
    for name, eig, zeta in modes:
        status = "stable" if eig.real < 0 else "UNSTABLE"
        print(f"  {name:<20} {eig.real:>10.4f} {eig.imag:>10.4f} {zeta:>8.4f}  {status}")

    print("\nRunning aero_score()...")
    score = aero_score(cruise_speed, stab)
    print(f"\n── Aero score (stable baseline) ──────────────────────────")
    print(f"  Lap time: {score.lap_time:.2f} s  |  Can fly: {score.can_fly}  |  Penalty: {score.penalty:.4f}")
    print(f"    ↳ SM={score.penalty_static_margin:.4f}  Cma={score.penalty_longitudinal:.4f}"
          f"  Cnb={score.penalty_directional:.4f}  Spiral={score.penalty_spiral:.4f}")

    assert score.lap_time > 0
    assert score.can_fly,       "Default DesignVector should be flyable"
    assert score.penalty == 0.0

    print("\n  [PASS] Stable baseline assertions passed.")
    return score


# ──────────────────────────────────────────────────────────────────────────
# Test 2: unstable design (static gates fail)
# ──────────────────────────────────────────────────────────────────────────

def test_unstable_design(stable_score: AeroScore) -> AeroScore:
    """Cma>0, SM<0 → can_fly=False, penalty>0. Spiral intentionally passes."""
    print("\n── Unstable design test ──────────────────────────────────")

    bad_stab = StabilityResult(
        Cma=+0.35,           # UNSTABLE: pitch diverges
        Cnb=+0.18,           # OK
        Clb=-0.023,          # OK
        static_margin=-0.15, # UNSTABLE: CG behind NP
        x_np=0.123,
        stall_speed=10.0,    # representative V_stall [m/s] for penalty test
        phugoid_eigenvalue=complex(-0.02, 0.15),    phugoid_damping_ratio=0.13,
        short_period_eigenvalue=complex(-1.8, 3.2), short_period_damping_ratio=0.49,
        roll_subsidence_eigenvalue=complex(-9.1, 0.), roll_subsidence_damping_ratio=1.0,
        dutch_roll_eigenvalue=complex(-0.3, 1.8),   dutch_roll_damping_ratio=0.16,
        spiral_eigenvalue=complex(+0.05, 0.),       # passes: dt=13.9 s > 4 s
        spiral_damping_ratio=-0.05,
    )

    score = aero_score(CRUISE_SPEED, bad_stab)
    print(f"  Cma={bad_stab.Cma:+.2f}  SM={bad_stab.static_margin:+.2f}  "
          f"spiral_λ={bad_stab.spiral_eigenvalue.real:+.3f} s⁻¹")
    print(f"  Can fly:  {score.can_fly}")
    print(f"  Penalty:  {score.penalty:.4f} / 10.0")
    print(f"    ↳ SM={score.penalty_static_margin:.4f}  Cma={score.penalty_longitudinal:.4f}"
          f"  Cnb={score.penalty_directional:.4f}  Spiral={score.penalty_spiral:.4f}")

    print("\n  SM vs. penalty (Cma=+0.35, spiral OK):")
    print(f"  {'SM':>8} {'can_fly':>9} {'penalty':>9}")
    for sm in [0.05, 0.00, -0.05, -0.15, -0.30]:
        probe = StabilityResult(
            Cma=bad_stab.Cma, Cnb=bad_stab.Cnb, Clb=bad_stab.Clb,
            static_margin=sm, x_np=bad_stab.x_np,
            stall_speed=bad_stab.stall_speed,
            phugoid_eigenvalue=bad_stab.phugoid_eigenvalue,
            phugoid_damping_ratio=bad_stab.phugoid_damping_ratio,
            short_period_eigenvalue=bad_stab.short_period_eigenvalue,
            short_period_damping_ratio=bad_stab.short_period_damping_ratio,
            roll_subsidence_eigenvalue=bad_stab.roll_subsidence_eigenvalue,
            roll_subsidence_damping_ratio=bad_stab.roll_subsidence_damping_ratio,
            dutch_roll_eigenvalue=bad_stab.dutch_roll_eigenvalue,
            dutch_roll_damping_ratio=bad_stab.dutch_roll_damping_ratio,
            spiral_eigenvalue=bad_stab.spiral_eigenvalue,
            spiral_damping_ratio=bad_stab.spiral_damping_ratio,
        )
        p = aero_score(CRUISE_SPEED, probe)
        print(f"  {sm:>+8.2f} {str(p.can_fly):>9} {p.penalty:>9.4f}")

    assert not score.can_fly
    assert score.penalty > 0.0
    assert score.penalty <= 10.0
    assert score.penalty_spiral == 0.0, "Spiral passes in this test"
    # Lap time depends on cruise speed and stall speed.
    # Compare against the analytical value for CRUISE_SPEED + bad_stab.stall_speed.
    expected_lt = _compute_lap_time(CRUISE_SPEED, bad_stab.stall_speed)
    assert abs(score.lap_time - expected_lt) < 1e-9, \
        f"Lap time should be {expected_lt:.3f} s at V={CRUISE_SPEED} m/s"

    print("\n  [PASS] Unstable design assertions passed.")
    return score


# ──────────────────────────────────────────────────────────────────────────
# Test 2b: spiral criterion
# ──────────────────────────────────────────────────────────────────────────

def test_spiral_criterion() -> None:
    """Statically stable but spiral doubling time < 4 s → can_fly=False."""
    print("\n── Spiral criterion test ─────────────────────────────────")

    def _make_stab(spiral_lam):
        return StabilityResult(
            Cma=-0.50, Cnb=+0.18, Clb=-0.023,
            static_margin=+0.10, x_np=0.123,
            stall_speed=10.0,   # representative V_stall for spiral-criterion test
            phugoid_eigenvalue=complex(-0.02, 0.15),    phugoid_damping_ratio=0.13,
            short_period_eigenvalue=complex(-1.8, 3.2), short_period_damping_ratio=0.49,
            roll_subsidence_eigenvalue=complex(-9.1, 0.), roll_subsidence_damping_ratio=1.0,
            dutch_roll_eigenvalue=complex(-0.3, 1.8),   dutch_roll_damping_ratio=0.16,
            spiral_eigenvalue=complex(spiral_lam, 0.),
            spiral_damping_ratio=float(-np.sign(spiral_lam)) if spiral_lam != 0 else 0.,
        )

    # Fast spiral: λ=0.35 s⁻¹ → doubling time ≈ 1.98 s < 4 s → FAIL
    fast = _make_stab(0.35)
    fast_score = aero_score(CRUISE_SPEED, fast)
    dt_fast = np.log(2) / 0.35
    print(f"  λ=+0.35 s⁻¹ → dt={dt_fast:.2f} s  can_fly={fast_score.can_fly}  "
          f"penalty={fast_score.penalty:.4f}  p_spiral={fast_score.penalty_spiral:.4f}")

    assert not fast_score.can_fly,             "Fast spiral must prevent flying"
    assert fast_score.penalty_spiral > 0,      "Fast spiral must yield spiral penalty"
    assert fast_score.penalty_static_margin == 0., "SM is fine"
    assert fast_score.penalty_longitudinal  == 0., "Cma is fine"
    assert fast_score.penalty_directional   == 0., "Cnb is fine"

    # Borderline: λ exactly at threshold (ln(2)/4) → passes
    border = _make_stab(SPIRAL_RATE_MAX)
    border_score = aero_score(CRUISE_SPEED, border)
    print(f"  λ={SPIRAL_RATE_MAX:+.4f} s⁻¹ (threshold) → can_fly={border_score.can_fly}")
    assert border_score.can_fly, "Design exactly at threshold should pass"

    print(f"\n  Spiral growth rate sensitivity (all static gates pass):")
    print(f"  {'λ (s⁻¹)':>10} {'dt (s)':>8} {'can_fly':>9} {'penalty':>9}")
    for lam in [0.0, 0.10, SPIRAL_RATE_MAX, 0.20, 0.35, 0.50]:
        probe_score = aero_score(CRUISE_SPEED, _make_stab(lam))
        dt_str = f"{np.log(2)/lam:.2f}" if lam > 1e-9 else "∞"
        print(f"  {lam:>10.4f} {dt_str:>8} {str(probe_score.can_fly):>9}"
              f" {probe_score.penalty:>9.4f}")

    print("\n  [PASS] Spiral criterion assertions passed.")


# ──────────────────────────────────────────────────────────────────────────
# Test 3: DF1 real parameters (XFLR5 geometry + report CG/weight/speed)
# ──────────────────────────────────────────────────────────────────────────

def test_df1_real_parameters() -> None:
    """
    Duck Force One with XFLR5 geometry and XFLR5-computed inertia.

    Geometry source: Mission 2/3 Iteration 2.xml
      tail_arm = 0.791 m (wing_QC→HT_QC; M2 file: elevator LE=0.830, chord=0.152)
      [M3 tail arm is 0.806 m — difference is small; same DesignVector used for both]

    CG/weight/speed source: design report Table 5.3.1 / 5.5.1 (flight-validated)

    Inertia source: computed from XFLR5 point masses + wing/elevator span correction.
      M2: about XFLR5 CG (≈ report CG, difference < 4 mm)
      M3: about report CG via parallel-axis shift from XFLR5 mass-model CG
    """
    IN_TO_M  = 0.0254
    LBF_TO_N = 4.44822
    FTS_TO_MS = 0.3048

    # ── XFLR5 tail arm (M2 geometry; M3 differs by 15 mm → same dv for both) ──
    # elevator QC = 0.830 + 0.25*0.152 = 0.868 m from wing LE
    # wing QC     = 0.25 * 0.307       = 0.077 m from wing LE
    # tail_arm    = 0.868 - 0.077      = 0.791 m
    TAIL_ARM_XFLR5 = 0.791  # m

    dv = DesignVector(tail_arm=TAIL_ARM_XFLR5)

    print("\n" + "=" * 62)
    print("=== DF1 Real Parameters Test (XFLR5 geometry) ===")
    print("=" * 62)
    print(f"Wing:     {dv.wing_span:.3f} m × {dv.wing_chord:.3f} m  (AR {dv.wing_span/dv.wing_chord:.2f})")
    print(f"HT:       {dv.hstab_span*39.37:.2f} in span × {dv.hstab_chord*39.37:.2f} in chord "
          f"(XFLR5: 17.24 in × 5.98 in)")
    print(f"VT:       {dv.vstab_span*39.37:.2f} in span × {dv.vstab_chord*39.37:.2f} in chord "
          f"(XFLR5: 5.24 in × 5.74 in)")
    print(f"tail_arm: {dv.tail_arm:.3f} m  (previously 0.845 m)")

    # ── M2 inertia from XFLR5 ─────────────────────────────────────────────
    m2_total, m2_xflr5_cg, m2_inertia = _xflr5_inertia(_M2_MASSES)
    print(f"\nM2 XFLR5 mass model:")
    print(f"  Total mass:   {m2_total:.3f} kg  (report: 7.23 lbf = 3.279 kg)")
    print(f"  XFLR5 CG:    x={m2_xflr5_cg[0]:.4f} m,  z={m2_xflr5_cg[2]:.4f} m")
    print(f"  Report CG:   x=0.0617 m  (2.43 in),  z=-0.0569 m  (2.24 in below LE)")
    print(f"  Inertia:     Ixx={m2_inertia[0,0]:.3f}  Iyy={m2_inertia[1,1]:.3f}  "
          f"Izz={m2_inertia[2,2]:.3f}  kg·m²")
    print(f"  (prev est:   Ixx=0.120  Iyy=0.180  Izz=0.280)")

    # ── M3 inertia from XFLR5 ─────────────────────────────────────────────
    # XFLR5 M3 mass list CG (0.118 m) differs from report (0.085 m) because
    # the file includes a "M3 ballast" that is not on the real aircraft.
    # We exclude the ballast and compute inertia about the report CG.
    CG_M3_REPORT = [3.33 * IN_TO_M, 0., -2.13 * IN_TO_M]  # [0.0846, 0, -0.0541]
    m3_total, m3_xflr5_cg, m3_inertia = _xflr5_inertia(_M3_MASSES, cg_override=CG_M3_REPORT)
    print(f"\nM3 XFLR5 mass model:")
    print(f"  Total mass (no ballast): {m3_total:.3f} kg  (report: 7.37 lbf = 3.341 kg)")
    print(f"  Inertia about report CG: Ixx={m3_inertia[0,0]:.3f}  Iyy={m3_inertia[1,1]:.3f}  "
          f"Izz={m3_inertia[2,2]:.3f}  kg·m²")
    print(f"  (Iyy is larger than M2 because banner at x=0.518 m is far aft of CG)")

    # ══════════════════════════════════════════════════════════════════════
    # Mission 2
    # ══════════════════════════════════════════════════════════════════════
    CG_M2     = [2.43 * IN_TO_M, 0., -2.24 * IN_TO_M]  # report Table 5.3.1
    WEIGHT_M2 = 7.23 * LBF_TO_N                          # 32.15 N
    V_M2      = 90.88 * FTS_TO_MS                         # 27.70 m/s
    ALPHA_M2  = 1.0   # deg (CL_trim ≈ 0.19 → ≈ 1° for NACA 2412 at AR 3.85)

    print(f"\n── Mission 2 ─────────────────────────────────────────────")
    print(f"  V={V_M2:.2f} m/s  W={WEIGHT_M2:.2f} N  "
          f"CG x={CG_M2[0]:.4f} m  z={CG_M2[2]:.4f} m")

    cc_m2  = CruiseCondition(asb.OperatingPoint(velocity=V_M2, alpha=ALPHA_M2), throttle=0.65)
    stab_m2 = stability_analysis(dv, cc_m2, _AERO_STUB, CG_M2, m2_inertia, WEIGHT_M2)

    spiral_λ_m2 = stab_m2.spiral_eigenvalue.real
    spiral_dt_m2 = np.log(2) / spiral_λ_m2 if spiral_λ_m2 > 1e-9 else float('inf')
    dt_str_m2 = f"{spiral_dt_m2:.1f} s" if np.isfinite(spiral_dt_m2) else "∞"

    print(f"  Cma:    {stab_m2.Cma:+.4f}  ({'stable' if stab_m2.Cma < 0 else 'UNSTABLE'})")
    print(f"  Cnb:    {stab_m2.Cnb:+.4f}  ({'stable' if stab_m2.Cnb > 0 else 'UNSTABLE'})")
    print(f"  SM:     {stab_m2.static_margin:+.4f}  ({stab_m2.static_margin*100:.1f}% MAC)")
    print(f"  Spiral: λ={spiral_λ_m2:+.4f} s⁻¹  dt={dt_str_m2}  "
          f"({'OK ≥4 s' if spiral_λ_m2 <= SPIRAL_RATE_MAX else 'FAIL <4 s'})")

    score_m2 = aero_score(V_M2, stab_m2)
    print(f"  → can_fly={score_m2.can_fly}  penalty={score_m2.penalty:.4f}"
          f"  lap_time={score_m2.lap_time:.2f} s  (report: 37.10 s)")

    assert score_m2.can_fly,              "DF1 M2 should be flyable"
    assert score_m2.penalty == 0.0
    assert score_m2.penalty_spiral == 0.0, "M2 spiral should pass"

    # ══════════════════════════════════════════════════════════════════════
    # Mission 3
    # ══════════════════════════════════════════════════════════════════════
    WEIGHT_M3 = 7.37 * LBF_TO_N   # 32.77 N
    V_M3      = 52.48 * FTS_TO_MS  # 15.99 m/s
    ALPHA_M3  = 6.0   # deg (CL_trim ≈ 0.58)

    print(f"\n── Mission 3 ─────────────────────────────────────────────")
    print(f"  V={V_M3:.2f} m/s  W={WEIGHT_M3:.2f} N  "
          f"CG x={CG_M3_REPORT[0]:.4f} m  z={CG_M3_REPORT[2]:.4f} m")

    cc_m3  = CruiseCondition(asb.OperatingPoint(velocity=V_M3, alpha=ALPHA_M3), throttle=0.80)
    stab_m3 = stability_analysis(dv, cc_m3, _AERO_STUB, CG_M3_REPORT, m3_inertia, WEIGHT_M3)

    spiral_λ_m3 = stab_m3.spiral_eigenvalue.real
    spiral_dt_m3 = np.log(2) / spiral_λ_m3 if spiral_λ_m3 > 1e-9 else float('inf')
    dt_str_m3 = f"{spiral_dt_m3:.1f} s" if np.isfinite(spiral_dt_m3) else "∞"

    print(f"  Cma:    {stab_m3.Cma:+.4f}  ({'stable' if stab_m3.Cma < 0 else 'UNSTABLE'})")
    print(f"  Cnb:    {stab_m3.Cnb:+.4f}  ({'stable' if stab_m3.Cnb > 0 else 'UNSTABLE'})")
    print(f"  SM:     {stab_m3.static_margin:+.4f}  ({stab_m3.static_margin*100:.1f}% MAC)")
    print(f"  Spiral: λ={spiral_λ_m3:+.4f} s⁻¹  dt={dt_str_m3}  "
          f"({'OK ≥4 s' if spiral_λ_m3 <= SPIRAL_RATE_MAX else 'FAIL <4 s'})")

    score_m3 = aero_score(V_M3, stab_m3)
    print(f"  → can_fly={score_m3.can_fly}  penalty={score_m3.penalty:.4f}"
          f"  lap_time={score_m3.lap_time:.2f} s  (report: 59.90 s)")

    assert score_m3.can_fly,              "DF1 M3 should be flyable"
    assert score_m3.penalty == 0.0
    assert score_m3.penalty_spiral == 0.0, "M3 spiral should pass"

    # Sanity: M3 CG is further aft → smaller static margin
    assert stab_m3.static_margin < stab_m2.static_margin, \
        f"M3 SM ({stab_m3.static_margin:.3f}) should be < M2 SM ({stab_m2.static_margin:.3f})"

    print(f"\n── Summary ───────────────────────────────────────────────")
    print(f"  {'':10} {'V (m/s)':>8} {'SM%MAC':>8} {'Iyy':>7} "
          f"{'spiral_dt':>10} {'can_fly':>9} {'lap_t(s)':>9}")
    for label, v, stab, score, iyy, dt_str in [
        ("M2", V_M2, stab_m2, score_m2, m2_inertia[1,1], dt_str_m2),
        ("M3", V_M3, stab_m3, score_m3, m3_inertia[1,1], dt_str_m3),
    ]:
        print(f"  {label:<10} {v:>8.2f} {stab.static_margin*100:>7.1f}% {iyy:>7.3f} "
              f"{dt_str:>10} {str(score.can_fly):>9} {score.lap_time:>9.2f}")

    print("\n  [PASS] DF1 real parameter assertions passed.")


# ──────────────────────────────────────────────────────────────────────────

def run_all_tests() -> None:
    stable_score = test_stable_design()
    test_unstable_design(stable_score)
    test_spiral_criterion()
    test_df1_real_parameters()
    print("\n[DONE] All aero scoring tests passed.")


if __name__ == "__main__":
    run_all_tests()
