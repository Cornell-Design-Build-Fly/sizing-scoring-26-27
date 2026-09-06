"""Retest deployed derivatives against fresh full analysis; preserve original study."""
import csv
import json
from pathlib import Path
import numpy as np
import aerosandbox as asb
from src.aero.utils import require_scalar
from src.aero.custom_classes import CruiseCondition
from src.aero.stability_analysis_coarse import estimate_stability_derivatives
from src.testing.calibrate_static_derivatives import TARGETS, dump_csv, metrics, predict
from src.vectors import DesignVector, ASBDesignVector

ROOT = Path('data_dump/static_calibration_20260906')

def main():
    out = ROOT / 'implemented_retest'
    out.mkdir(exist_ok=True)
    coef = json.loads((ROOT/'under_55lb/summary.json').read_text())['training_only_coefficients']['coupled_enriched']
    report = {}
    for name, source in [('under_55lb', ROOT/'under_55lb'), ('broad', ROOT)]:
        designs = json.loads((source/'designs.json').read_text())
        with (source/'raw_results.csv').open() as stream:
            rows = [{k: v if k == 'split' else float(v) for k,v in r.items()} for r in csv.DictReader(stream)]
        delta = 0.
        for i, row in enumerate(rows):
            cfg = designs[int(row['design_id'])]
            d = DesignVector(**{k:v for k,v in cfg['design'].items() if DesignVector.__dataclass_fields__[k].init})
            cg = list(cfg['cg_center']); cg[0] = row['cg_x']
            op = asb.OperatingPoint(velocity=row['velocity'], alpha=row['alpha'])
            cc = CruiseCondition(op, 10., True, elevator_deflection=row['elevator'])
            mp = asb.MassProperties(mass=cfg['mass'], x_cg=cg[0], y_cg=cg[1], z_cg=cg[2], Ixx=1,Iyy=1,Izz=1)
            actual = estimate_stability_derivatives(d,cc,mp)
            expected = predict([row], 'coupled_enriched', coef)
            plane = ASBDesignVector.from_design_vector(d).make_airplane(elevator_deflection=row['elevator'])
            full = asb.AeroBuildup(plane,op,xyz_ref=cg,include_wave_drag=False).run_with_stability_derivatives(p=False,q=False,r=False)
            for key in TARGETS:
                np.testing.assert_allclose(actual[key], expected[key][0], atol=1e-12, rtol=1e-12)
                delta = max(delta, abs(require_scalar(full[key])-row['full_'+key]))
                row['new_'+key] = float(actual[key])
                row['full_'+key] = require_scalar(full[key])
            if (i+1)%30 == 0: print(f'{name}: {i+1}/150', flush=True)
        dump_csv(out/(name+'_results.csv'), rows)
        report[name] = {'cases':len(rows), 'max_full_rerun_difference':delta}
        for split in ['train','holdout']:
            subset = [r for r in rows if r['split']==split]
            report[name][split] = {label:metrics(subset,{k:np.array([r[prefix+k] for r in subset]) for k in TARGETS}) for label,prefix in [('before','old_'),('after','new_')]}
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__ == '__main__': main()

