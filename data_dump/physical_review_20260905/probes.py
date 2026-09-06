"""Read-only model probes. Run from repo root with python -m data_dump.physical_review_20260905.probes."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np
import aerosandbox as asb

from src.vectors import DesignVector, ParameterVector, ASBDesignVector
from src.mech.main_mech import evaluate_mechanical_module
from src.aero.custom_classes import CruiseCondition
from src.aero.stability_analysis_coarse import estimate_stability_derivatives
from src.aero.cruise_analysis_fast import cruise_analysis_fast
from src.aero.flight_profile import compute_flight_profile, DEFAULT_FLIGHT_PROFILE_CONFIG, _drag_estimate
from src.prop.continuous_prop_database import load_default_continuous_prop_database
from src.prop.prop_cruise_values import solve_cruise_samples
from src.prop.prop_helper_functions import make_motor_from_design, make_battery_from_design
from src.prop.prop_classes import DEFAULT_VELOCITIES_MPS
from src.prop.main_prop import prop_main


def run():
    out = {}
    pv = ParameterVector()
    dv = DesignVector()
    mech = evaluate_mechanical_module(dv, parameter_vector=pv)
    out['baseline_mechanics'] = {}
    for name, mp in mech.missions.items():
        out['baseline_mechanics'][name] = dict(mass=mp.total_mass_kg, cg=mp.cg_m, static_margin=mp.static_margin,
                                               inertia_eigenvalues=np.linalg.eigvalsh(mp.inertia_tensor_kg_m2).tolist())
    fuselage = next(i for i in mech.all_items if i.name == 'Fuselage structure')
    out['fuselage'] = dict(mech_front=float(fuselage.position_m[0]-fuselage.dimensions_m[0]/2),
                           mech_back=float(fuselage.position_m[0]+fuselage.dimensions_m[0]/2),
                           aero_nose=-dv.nose_length, aero_taper_start=dv.wing_chord,
                           width=mech.resolved_fuselage_width_m,height=mech.resolved_fuselage_height_m)
    resolved = replace(dv, fuselage_width=mech.resolved_fuselage_width_m, fuselage_height=mech.resolved_fuselage_height_m)
    from src.aero.tow_line_model import tow_line_force_components
    from src.aero.flight_profile import _lap_phases
    hook = next(i for i in mech.all_items if i.category == 'release_mechanism')
    backward, downward, _ = tow_line_force_components(dv,pv,30)
    out['tow_moment_if_release_is_attachment'] = np.cross(
        hook.position_m-np.asarray(mech.for_mission('M1').cg_m),[float(backward),0,-float(downward)]).tolist()
    out['ground_clearance'] = []
    for count in (0,3,10):
        result=evaluate_mechanical_module(replace(dv,extra_shipping_containers=count))
        f=next(i for i in result.all_items if i.name=='Fuselage structure')
        gear=next(i for i in result.all_items if i.name=='Landing gear')
        out['ground_clearance'].append(dict(extra_containers=count,
            body_bottom=float(f.position_m[2]-f.dimensions_m[2]/2),
            gear_bottom=float(gear.position_m[2]-gear.dimensions_m[2]/2)))
    out['turn_example'] = dict(lap_time_without_thrust_check=_lap_phases(15,10,pv,DEFAULT_FLIGHT_PROFILE_CONFIG)[0],
        level_drag=float(_drag_estimate(dv,pv,15,4*pv.gravity,sensor_deployed=False,config=DEFAULT_FLIGHT_PROFILE_CONFIG)),
        bank35_drag=float(_drag_estimate(dv,pv,15,4*pv.gravity/np.cos(np.deg2rad(35)),sensor_deployed=False,config=DEFAULT_FLIGHT_PROFILE_CONFIG)))
    out['neutral_points'] = []
    for x in (0.02, 0.08, 0.14, 0.20):
        mp = asb.MassProperties(mass=4, x_cg=x, Ixx=.2,Iyy=.3,Izz=.4)
        cc = CruiseCondition(asb.OperatingPoint(velocity=25,alpha=4),10,True)
        est = estimate_stability_derivatives(resolved,cc,mp)
        full = asb.AeroBuildup(airplane=ASBDesignVector.from_design_vector(resolved).make_airplane(),
                              op_point=cc.operating_point,xyz_ref=[x,0,0],include_wave_drag=False).run_with_stability_derivatives()
        out['neutral_points'].append(dict(cg=x,coarse_np=est['x_np'],coarse_cnb=est['Cnb'],
                                         full_np=float(np.asarray(full['x_np']).item()),full_cnb=float(np.asarray(full['Cnb']).item())))
    # Identical flight inputs: altitude only affects modeled climb, never landing.
    out['altitude_profiles'] = []
    for height in (30.48,60.96):
        profile=compute_flight_profile(22,10,dv,pv,(-.02,-.5,120),4,1,(0,0,1000),
                                      replace(DEFAULT_FLIGHT_PROFILE_CONFIG,first_turn_altitude_m=height))
        out['altitude_profiles'].append(dict(height=height,feasible=bool(profile.feasible),
                                            landing_time=profile.landing_time_s,climb_time=profile.climb_time_s))
    print('Geometry and stability probes complete.',flush=True)
    db=load_default_continuous_prop_database()
    out['baseline_prop_fits'] = {}
    for mission in (1,3):
        thrust,endurance=prop_main(resolved,pv,mission,prop_database=db)
        out['baseline_prop_fits'][mission] = dict(thrust=list(thrust),endurance=list(endurance))
    out['baseline_cruise'] = {}
    for mission in (1,2,3):
        props=mech.for_mission('M1' if mission==3 else f'M{mission}')
        thrust,endurance=prop_main(resolved,pv,mission,prop_database=db)
        cc=cruise_analysis_fast(resolved,pv,thrust,props.cg_m,props.total_mass_kg,mission)
        out['baseline_cruise'][mission] = dict(converged=bool(cc.converged),speed=float(cc.operating_point.velocity),stall=cc.stall_speed)
    from src.main import main
    net, breakdown, details = main(dv,pv,prop_database=db,return_details=True)
    out['baseline_main'] = dict(net_score=net,breakdown=breakdown,
        missions={name:dict(can_fly=bool(s.can_fly),reason=s.flight_profile_reason,penalty=s.penalty) for name,s in details.items()})
    from src.opt.topline_opt import _pd_ratio, _ducks_per_puck
    out['optimizer'] = dict(intended_pd=dv.prop_pitch_in/dv.prop_diameter_in,actual_constraint_value=_pd_ratio(dv.to_array()))
    try:
        _ducks_per_puck(dv.to_array())
    except ValueError as exc:
        out['optimizer']['legacy_payload_error'] = str(exc)
    # In-bound propeller/motor designs with the optimizer's intended P/D range.
    out['prop_energy_violations'] = []
    out['fit_extrapolation'] = []
    rng=np.random.default_rng(20260905)
    for k in range(35):
        d=float(rng.uniform(10,25)); p=float(rng.uniform(max(5,.4*d), min(18,.8*d)))
        design=replace(dv,prop_diameter_in=d,prop_pitch_in=p,motor_kv=float(rng.uniform(200,500)),
                       motor_max_power=float(rng.uniform(1000,3000)),cruise_throttle=float(rng.uniform(.5,1)))
        motor=make_motor_from_design(design,pv); battery=make_battery_from_design(design,pv)
        velocities=np.array([.01,9.5,19,28.35,35,45,50])
        res=solve_cruise_samples(d,p,velocities,motor,battery,100,design.cruise_throttle,db)
        t,q=db.evaluate(d,p,velocities*2.2369,res.selected_rpm)
        shaft=np.asarray(q)*res.selected_rpm*2*np.pi/60
        mask=(~res.failed_mask)&(res.thrust_samples_n>0)&((shaft<=0)|(res.thrust_samples_n*velocities>shaft*1.01))
        for j in np.flatnonzero(mask):
            out['prop_energy_violations'].append(dict(design=design.to_array().tolist(),velocity=float(velocities[j]),
                rpm=float(res.selected_rpm[j]),thrust=float(res.thrust_samples_n[j]),torque=float(q[j]),
                useful_power=float(res.thrust_samples_n[j]*velocities[j]),shaft_power=float(shaft[j])))
        if not np.any(res.failed_mask[:4]):
            fit=np.polyfit(DEFAULT_VELOCITIES_MPS,res.thrust_samples_n[:4],2)
            pred=float(np.polyval(fit,50)); actual=float(res.thrust_samples_n[-1])
            if pred>0 and abs(pred-actual)>max(2,abs(actual)*.3):
                out['fit_extrapolation'].append(dict(design=design.to_array().tolist(),predicted_50=pred,direct_50=actual))
    # Airfoil changes have no effect on active fast trim.
    out['airfoil_invariance'] = []
    for foil in ('naca0012','naca2412','naca4412'):
        cc=cruise_analysis_fast(replace(resolved,wing_airfoil=foil),pv,(-.02,-.5,40),(.08,0,-.05),4,1)
        out['airfoil_invariance'].append(dict(airfoil=foil,converged=bool(cc.converged),speed=float(cc.operating_point.velocity),stall=cc.stall_speed))
    path=Path(__file__).with_name('probe_results.json')
    path.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2),flush=True)


if __name__ == '__main__':
    run()
