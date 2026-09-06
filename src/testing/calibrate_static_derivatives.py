"""150-case static derivative study; does not alter production calibration.

Run: python -m src.testing.calibrate_static_derivatives
Refit saved evaluations: python -m src.testing.calibrate_static_derivatives --reuse
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import aerosandbox as asb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import lsq_linear
from scipy.stats import qmc

from src.aero.custom_classes import CruiseCondition
from src.aero.stability_analysis_coarse import estimate_stability_derivatives
from src.aero.utils import require_scalar
from src.mech.main_mech import evaluate_mechanical_module
from src.vectors import DesignVector, ASBDesignVector, ParameterVector, MIN_SENSOR_WEIGHT_KG, MAX_SENSOR_WEIGHT_KG

OUT = Path('data_dump/static_calibration_20260906')
SEED = 20260906
TARGETS = ('CLa', 'Cma', 'Cnb')


def dump_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_designs(practical=False):
    samples = qmc.LatinHypercube(d=9, seed=SEED+(1 if practical else 0)).random(2000 if practical else 400)
    accepted = []
    rejected = []
    for attempt, u in enumerate(samples):
        d = DesignVector(
            wing_span=.914 + u[0]*(1.524-.914), wing_chord=.12+u[1]*.28,
            tail_arm=.3+u[2]*.6, nose_length=.08+u[3]*.22,
            extra_shipping_containers=min(10,int(u[4]*11)),
            sensor_weight_kg=MIN_SENSOR_WEIGHT_KG+u[5]*(MAX_SENSOR_WEIGHT_KG-MIN_SENSOR_WEIGHT_KG))
        if not accepted:
            d=DesignVector()
        if d.tail_arm < d.wing_chord+.03:
            rejected.append(dict(attempt=attempt, reason='Less than 30 mm wing-to-tail LE clearance beyond wing TE'))
            continue
        try:
            mech=evaluate_mechanical_module(d,parameter_vector=ParameterVector())
        except (ValueError,RuntimeError) as exc:
            rejected.append(dict(attempt=attempt,reason=str(exc)))
            continue
        resolved=replace(d,fuselage_width=mech.resolved_fuselage_width_m,fuselage_height=mech.resolved_fuselage_height_m)
        props=mech.for_mission('M2')
        if practical and props.total_mass_kg>55*.45359237:
            rejected.append(dict(attempt=attempt,reason='Above historical 55 lb M2 limit'))
            continue
        accepted.append(dict(design_id=len(accepted),design=asdict(resolved),
            cg_center=list(props.cg_m),mass=props.total_mass_kg,
            velocity=20+22*u[6],alpha=-2+12*u[7],elevator=-8+16*u[8]))
        if len(accepted)==50:
            break
    if len(accepted)!=50:
        raise RuntimeError('Could not sample 50 feasible mechanical geometries')
    return accepted,rejected


def collect(practical=False):
    designs,rejected=make_designs(practical)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'designs.json').write_text(json.dumps(designs,indent=2))
    (OUT/'rejected_samples.json').write_text(json.dumps(rejected,indent=2))
    rng=np.random.default_rng(SEED)
    holdout=set(rng.permutation(50)[:15].tolist())
    rows=[]
    for cfg in designs:
        d=DesignVector(**{k:v for k,v in cfg['design'].items() if DesignVector.__dataclass_fields__[k].init})
        plane=ASBDesignVector.from_design_vector(d).make_airplane(elevator_deflection=cfg['elevator'])
        fuselage=plane.fuselages[0]
        xs=np.array([require_scalar(x.xyz_c[0]) for x in fuselage.xsecs])
        areas=np.array([require_scalar(x.xsec_area()) for x in fuselage.xsecs])
        xb=float(np.trapezoid(xs*areas,xs)/np.trapezoid(areas,xs))
        op=asb.OperatingPoint(velocity=cfg['velocity'],alpha=cfg['alpha'])
        for offset in (-.20,0,.20):
            cg=cfg['cg_center'].copy();cg[0]+=offset*d.wing_chord
            cc=CruiseCondition(op,stall_speed=10,converged=True,elevator_deflection=cfg['elevator'])
            mp=asb.MassProperties(mass=cfg['mass'],x_cg=cg[0],y_cg=cg[1],z_cg=cg[2],Ixx=1,Iyy=1,Izz=1)
            t=perf_counter()
            coarse=estimate_stability_derivatives(d,cc,mp)
            coarse_time=perf_counter()-t
            t=perf_counter()
            full=asb.AeroBuildup(plane,op,xyz_ref=cg,include_wave_drag=False).run_with_stability_derivatives(p=False,q=False,r=False)
            full_time=perf_counter()-t
            row=dict(design_id=cfg['design_id'],split='holdout' if cfg['design_id'] in holdout else 'train',
                cg_offset_mac=offset,cg_x=cg[0],cg_z=cg[2],mass=cfg['mass'],velocity=cfg['velocity'],
                alpha=cfg['alpha'],elevator=cfg['elevator'],wing_span=d.wing_span,wing_chord=d.wing_chord,
                tail_arm=d.tail_arm,nose_length=d.nose_length,fuselage_width=d.fuselage_width,
                fuselage_height=d.fuselage_height,wing_area=d.wing_area,hstab_area=d.hstab_area,
                hstab_chord=d.hstab_chord,vstab_area=d.vstab_area,vstab_chord=d.vstab_chord,
                body_length=float(fuselage.length()),body_volume=float(fuselage.volume()),body_x=xb,
                wing_re=require_scalar(op.reynolds(d.wing_chord)),coarse_seconds=coarse_time,full_seconds=full_time)
            for key in (*TARGETS,'CYb','CDa','CL','CD'):
                if key in coarse:row['old_'+key]=require_scalar(coarse[key])
                row['full_'+key]=require_scalar(full[key])
            if not all(np.isfinite(row['full_'+key]) for key in TARGETS):
                raise RuntimeError(f'Nonfinite full-model result: {row}')
            rows.append(row)
        print(f"Evaluated {len(rows)}/150 matched cases",flush=True)
    dump_csv(OUT/'raw_results.csv',rows)
    return rows


def features(rows, enriched=False):
    """Shared nonnegative force contributions; moment arms use actual CG.

    Cma translation uses CLa*cos(alpha) as the low-angle normal-force slope
    approximation. All component gains are nonnegative. The body-volume
    pitching couple is destabilizing; the yaw couple is destabilizing.
    """
    lift=[];pitch=[];yaw=[]
    for r in rows:
        s,c,b=r['wing_area'],r['wing_chord'],r['wing_span']
        aw=2*np.pi/(1+2/(b*b/s))
        ah=.9*.85*(2*np.pi/(1+2/3))*r['hstab_area']/s
        av=.9*(2*np.pi/(1+2/.89))*r['vstab_area']/s
        a=np.deg2rad(r['alpha']);ca=np.cos(a)
        # Separate body crossflow force and inviscid volume moment.
        bf=r['fuselage_width']*r['body_length']/s*abs(np.sin(a))
        by=r['fuselage_height']*r['body_length']/s*abs(np.sin(a))
        pw=(r['cg_x']-.25*c)/c*ca
        ph=(r['cg_x']-r['tail_arm']-.25*r['hstab_chord'])/c*ca
        pb=(r['cg_x']-r['body_x'])/c*ca
        lf=[aw,ah,bf,0.]
        pf=[aw*pw,ah*ph,bf*pb,r['body_volume']/(s*c)]
        yf=[av*(r['tail_arm']+.25*r['vstab_chord']-r['cg_x'])/b,
            -r['body_volume']/(s*b),by*(r['body_x']-r['cg_x'])/b]
        if enriched:
            # Bounded alpha/Re shape terms shared by lift and its pitch moment.
            wa=max(0.,1-(r['alpha']/15)**2)
            ha=max(0.,1-((r['alpha']+.65*r['elevator'])/20)**2)
            wr=np.log1p(r['wing_re']/1e5)
            lf.extend([aw*wa,ah*ha,aw*wr])
            pf.extend([aw*wa*pw,ah*ha*ph,aw*wr*pw])
        lift.append(lf);pitch.append(pf);yaw.append(yf)
    return np.array(lift),np.array(pitch),np.array(yaw)


def original_features(rows):
    X={k:[] for k in TARGETS}
    for r in rows:
        s,c,b=r['wing_area'],r['wing_chord'],r['wing_span'];x=r['cg_x']
        aw=2*np.pi/(1+2/(b*b/s));ah=.9*.85*2*np.pi/(1+2/3)*r['hstab_area']/s
        lt=r['tail_arm']+.25*r['hstab_chord']-x
        cy=-.9*2*np.pi/(1+2/.89)*r['vstab_area']/s
        X['CLa'].append([1,aw+ah])
        X['Cma'].append([1,aw*(x-.25*c)/c-ah*lt/c])
        X['Cnb'].append([1,-cy*lt/b-.05,cy,b,c,r['alpha']])
    return {k:np.asarray(v) for k,v in X.items()}


def fit(rows, model):
    y={k:np.array([r['full_'+k] for r in rows]) for k in TARGETS}
    if model=='unrestricted_refit':
        X=original_features(rows)
        return {k:np.linalg.lstsq(X[k],y[k],rcond=None)[0].tolist() for k in TARGETS}
    if model in ('reference_current_lift','reference_refit_lift'):
        L,_,N=features(rows,True)
        liftcoef=None
        cla=np.array([r['old_CLa'] for r in rows])
        if model=='reference_refit_lift':
            liftcoef=lsq_linear(L,y['CLa'],bounds=(0,np.inf),tol=1e-12).x.tolist()
            cla=L@liftcoef
        P,dx=reference_features(rows)
        pitchcoef=lsq_linear(P,y['Cma']-cla*dx,
            bounds=([-np.inf,0,0,0],[np.inf,np.inf,np.inf,np.inf]),tol=1e-12)
        yawcoef=lsq_linear(N,y['Cnb'],bounds=(0,np.inf),tol=1e-12)
        return dict(lift=liftcoef,pitch_reference=pitchcoef.x.tolist(),yaw=yawcoef.x.tolist())
    L,M,N=features(rows,model=='coupled_enriched')
    # Fixed weights avoid choosing weights after viewing holdout performance.
    X=np.vstack([L/4,M]);target=np.concatenate([y['CLa']/4,y['Cma']])
    coef=lsq_linear(X,target,bounds=(0,np.inf),tol=1e-12,max_iter=1000)
    yawcoef=lsq_linear(N,y['Cnb'],bounds=(0,np.inf),tol=1e-12,max_iter=1000)
    if not coef.success or not yawcoef.success:raise RuntimeError('Bounded calibration failed')
    return dict(lift_pitch=coef.x.tolist(),yaw=yawcoef.x.tolist())


def reference_features(rows):
    P=[];dx=[]
    for r in rows:
        s,c=r['wing_area'],r['wing_chord'];a=np.deg2rad(r['alpha'])
        ah=.9*.85*2*np.pi/(1+2/3)*r['hstab_area']/s
        bf=r['fuselage_width']*r['body_length']/s*abs(np.sin(a))
        P.append([1,-ah*(r['tail_arm']+.25*r['hstab_chord']-.25*c)/c*np.cos(a),
            r['body_volume']/(s*c),bf*(.25*c-r['body_x'])/c*np.cos(a)])
        dx.append((r['cg_x']-.25*c)/c*np.cos(a))
    return np.array(P),np.array(dx)


def predict(rows,model,coef=None):
    if model=='current':return {k:np.array([r['old_'+k] for r in rows]) for k in TARGETS}
    if model=='unrestricted_refit':
        return {k:X@np.array(coef[k]) for k,X in original_features(rows).items()}
    if model in ('reference_current_lift','reference_refit_lift'):
        L,_,N=features(rows,True);P,dx=reference_features(rows)
        cla=np.array([r['old_CLa'] for r in rows]) if coef['lift'] is None else L@coef['lift']
        return dict(CLa=cla,Cma=P@coef['pitch_reference']+cla*dx,Cnb=N@coef['yaw'])
    L,M,N=features(rows,model=='coupled_enriched')
    return dict(CLa=L@coef['lift_pitch'],Cma=M@coef['lift_pitch'],Cnb=N@coef['yaw'])


def metrics(rows,pred):
    output={}
    for key in TARGETS:
        true=np.array([r['full_'+key] for r in rows]);error=pred[key]-true
        sign_true=true<0 if key=='Cma' else true>0
        sign_pred=pred[key]<0 if key=='Cma' else pred[key]>0
        output[key]=dict(mae=float(np.mean(abs(error))),rmse=float(np.sqrt(np.mean(error**2))),
            bias=float(np.mean(error)),p95_absolute_error=float(np.quantile(abs(error),.95)),
            max_absolute_error=float(np.max(abs(error))),correlation=float(np.corrcoef(true,pred[key])[0,1]),
            sign_agreement=float(np.mean(sign_true==sign_pred)),false_pass=int(np.sum(sign_pred&~sign_true)),
            false_rejection=int(np.sum(~sign_pred&sign_true)))
    true=-np.array([r['full_Cma'] for r in rows])/np.array([r['full_CLa'] for r in rows])
    sm=-pred['Cma']/pred['CLa']
    output['static_margin']=dict(mae=float(np.mean(abs(sm-true))),max_absolute_error=float(np.max(abs(sm-true))))
    ids=sorted(set(r['design_id'] for r in rows));slopes={k:[] for k in ('Cma','Cnb')}
    for i in ids:
        loc=[j for j,r in enumerate(rows) if r['design_id']==i]
        if len(loc)!=3:continue
        loc=sorted(loc,key=lambda j:rows[j]['cg_x'])
        for k in slopes:slopes[k].append((pred[k][loc[-1]]-pred[k][loc[0]])/(rows[loc[-1]]['cg_x']-rows[loc[0]]['cg_x']))
    output['cg_trends']=dict(pitch_slope_positive=int(np.sum(np.array(slopes['Cma'])>0)),
        yaw_slope_negative=int(np.sum(np.array(slopes['Cnb'])<0)),designs=len(ids))
    return output


def analyze(rows):
    train=[r for r in rows if r['split']=='train'];test=[r for r in rows if r['split']=='holdout']
    models=['current','unrestricted_refit','coupled_simple','coupled_enriched','reference_current_lift','reference_refit_lift']
    results={};coefs={}
    for model in models:
        coefs[model]=None if model=='current' else fit(train,model)
        results[model]=dict(train=metrics(train,predict(train,model,coefs[model])),
                           holdout=metrics(test,predict(test,model,coefs[model])))
    # Five grouped folds only within training set. Holdout never selects factors.
    ids=np.array(sorted(set(r['design_id'] for r in train)))
    folds=np.array_split(np.random.default_rng(SEED+1).permutation(ids),5)
    for model in models:
        cvpred={k:np.zeros(len(train)) for k in TARGETS}
        for fold in folds:
            training=[r for r in train if r['design_id'] not in fold]
            loc=[i for i,r in enumerate(train) if r['design_id'] in fold]
            valid=[train[i] for i in loc]
            cf=None if model=='current' else fit(training,model)
            p=predict(valid,model,cf)
            for k in TARGETS:cvpred[k][loc]=p[k]
        results[model]['grouped_cv']=metrics(train,cvpred)
    objective=lambda model:sum(results[model]['grouped_cv'][k]['rmse']/scale for k,scale in zip(TARGETS,(4,1,.1)))
    recommended=min(('coupled_simple','coupled_enriched','reference_current_lift','reference_refit_lift'),key=objective)
    prediction_rows=[]
    for i,r in enumerate(rows):
        v={k:r[k] for k in ('design_id','split','cg_offset_mac','cg_x','alpha','velocity')}
        for k in TARGETS:v['full_'+k]=r['full_'+k]
        for model in models:
            pred=predict([r],model,coefs[model])
            for k in TARGETS:v[model+'_'+k]=float(pred[k][0])
        prediction_rows.append(v)
    dump_csv(OUT/'predictions.csv',prediction_rows)
    summary=dict(seed=SEED,cases=len(rows),training_cases=len(train),holdout_cases=len(test),
        recommended_model=recommended,training_only_coefficients=coefs,
        all_data_refit_coefficients=fit(rows,recommended),results=results,
        model_scope='NACA2412; current resolved fuselage geometry; fixed matched operating points, not full cruise trim; airplane aero only, no tow',
        versions=dict(aerosandbox=asb.__version__,numpy=np.__version__),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),Path('src/vectors.py'),Path('src/aero/stability_analysis_coarse.py'))})
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    flat=[]
    for model in models:
        for split in ('train','grouped_cv','holdout'):
            for key in TARGETS:flat.append(dict(model=model,split=split,quantity=key,**results[model][split][key]))
    dump_csv(OUT/'metrics.csv',flat)
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for col,key in enumerate(TARGETS):
        true=np.array([r['full_'+key] for r in test])
        for row,model in enumerate(('current',recommended)):
            pred=predict(test,model,coefs[model])[key]
            ax=axes[row,col];ax.scatter(true,pred,s=18,alpha=.75)
            lo=min(true.min(),pred.min());hi=max(true.max(),pred.max())
            ax.plot([lo,hi],[lo,hi],color='gray',linestyle='--')
            ax.set(xlabel='Full AeroBuildup',ylabel=model,title=f"{key}: MAE {np.mean(abs(pred-true)):.4f}")
            ax.grid(alpha=.2)
    fig.suptitle('45 held-out cases from 15 unseen geometries; coefficients fitted on 105 cases')
    fig.savefig(OUT/'holdout_comparison.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(recommended=recommended,coefficients=coefs[recommended],
        holdout={m:results[m]['holdout'] for m in models}),indent=2),flush=True)


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--reuse',action='store_true')
    parser.add_argument('--practical',action='store_true',help='Additional independent geometry sample under historical 55 lb M2 limit')
    args=parser.parse_args()
    if args.practical:OUT=OUT/'under_55lb'
    if args.reuse:
        with (OUT/'raw_results.csv').open() as stream:
            rows=[{k:v if k=='split' else float(v) for k,v in r.items()} for r in csv.DictReader(stream)]
    else:rows=collect(args.practical)
    analyze(rows)


if __name__=='__main__':main()
