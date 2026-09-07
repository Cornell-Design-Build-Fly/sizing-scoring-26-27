import importlib,json,traceback
from pathlib import Path
from dataclasses import asdict
from time import perf_counter
import numpy as np
import aerosandbox as asb
from src.vectors import DesignVector,ParameterVector,ASBDesignVector
from src.aero.cruise_analysis import cruise_analysis
from src.aero.stability_analysis import stability_analysis
from src.aero.drag_model import sensor_drag_force
from src.aero.utils import require_scalar

out=Path('data_dump/full_aero_integration_check')
raw=Path(r'C:\Users\charl\.codex\attachments\fe20e269-2de1-4bcf-90bb-a5aa1537d88e\pasted-text.txt').read_text()
start=raw.index('{', raw.index(chr(34)+'optimizer_vector'+chr(34)))
payload={'optimizer_vector':json.JSONDecoder().raw_decode(raw[start:])[0]}
d=DesignVector(**{k:v for k,v in payload['optimizer_vector'].items() if k in DesignVector.__dataclass_fields__ and DesignVector.__dataclass_fields__[k].init})
module=importlib.import_module('src.main'); original=module.aero_main; captures={}
def capture(*args,**kwargs):
    captures[kwargs['mission']]=kwargs.copy()
    return original(*args,**kwargs)
module.aero_main=capture
results={'input':asdict(d)}
t=perf_counter()
try:
    baseline=module.main(d,ParameterVector(),return_details=True)
    results['integrated_baseline_score']=baseline[0]
except Exception:
    results['baseline_error']=traceback.format_exc()
finally: module.aero_main=original
results['baseline_seconds']=perf_counter()-t
results['missions']={}
def serialize(v):
    if hasattr(v,'tolist'):return v.tolist()
    return str(v)
def save(): (out/'results.json').write_text(json.dumps(results,indent=2,default=serialize))
save()
for mission,kw in captures.items():
    row={'inertial_mass':kw['mass'],'supported_mass':kw.get('supported_mass',kw['mass']),
         'cg':kw['cg'],'inertia_eigenvalues':np.linalg.eigvalsh(kw['inertia_matrix']).tolist(),
         'thrust_curve':kw['thrust_velocity']}
    results['missions'][str(mission)]=row
    print('full mission',mission,flush=True);t=perf_counter()
    try:
        dv=kw['design_vector'];pv=kw['parameter_vector']
        cc=cruise_analysis(dv,pv,kw['thrust_velocity'],kw['cg'],row['supported_mass'],mission)
        row['trim']=dict(converged=cc.converged,velocity=require_scalar(cc.operating_point.velocity),alpha=require_scalar(cc.operating_point.alpha),elevator=cc.elevator_deflection,stall_speed=cc.stall_speed)
        if cc.converged:
            mp=asb.MassProperties(mass=kw['mass'],x_cg=kw['cg'][0],y_cg=kw['cg'][1],z_cg=kw['cg'][2],
             Ixx=kw['inertia_matrix'][0,0],Iyy=kw['inertia_matrix'][1,1],Izz=kw['inertia_matrix'][2,2],Ixy=kw['inertia_matrix'][0,1],Iyz=kw['inertia_matrix'][1,2],Ixz=kw['inertia_matrix'][0,2])
            row['stability']=asdict(stability_analysis(dv,cc,mp))
            plane=ASBDesignVector.from_design_vector(dv).make_airplane(elevator_deflection=cc.elevator_deflection,tail_incidence=cc.tail_incidence)
            aero=asb.AeroBuildup(plane,cc.operating_point,xyz_ref=kw['cg'],include_wave_drag=False).run()
            weight=row['supported_mass']*pv.gravity
            drag=require_scalar(aero['D'])+(float(sensor_drag_force(dv,pv,row['trim']['velocity'])) if mission==3 else 0.)
            row['verified_residuals']={'lift':(require_scalar(aero['L'])-weight)/weight,
              'drag':(drag-float(np.polyval(kw['thrust_velocity'],row['trim']['velocity'])))/weight,
              'pitch':require_scalar(aero['m_b'])/(weight*dv.wing_chord)}
    except Exception:row['error']=traceback.format_exc()
    row['seconds']=perf_counter()-t;save();print(json.dumps(row,default=serialize),flush=True)
