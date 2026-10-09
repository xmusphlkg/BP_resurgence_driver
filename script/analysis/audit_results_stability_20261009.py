"""Supplement frozen results with direction, score, coverage and fresh-noise audits."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'script/src'),str(Path(__file__).parent)]
from run_ijid_revision_audits import regenerated_notifications
from pertussis_global_dynamics.bayes.model2c_surveillance_design import leave_one_country_out_predictions, feature_columns_for_package,roc_auc
OUT=ROOT/'outputs/results_stability_20261009';OUT.mkdir(exist_ok=True)
LP=ROOT/'outputs/model2c_surveillance_design/monitoring_packages_v4_source_preserved_20260908/latent_surveillance_signals.csv'
FP=ROOT/'outputs/agent_measurement_model_upgrade_20260911/observation_features_calibrated.csv'
HP=ROOT/'outputs/country_macro_revision_20260921/holdout_predictions.csv'
lat=pd.read_csv(LP);lat=lat[lat.scenario_id.isin(['matched_transmission_truth','matched_reporting_truth'])].copy()
assert len(lat)==1440 and lat.group_id.nunique()==720
positive='matched_transmission_truth';negative='matched_reporting_truth'
lat['three_year']=lat[[f'latent_severe_source_{y}' for y in (2023,2024,2025)]].sum(axis=1)
directions=[]
for window,col in [('2024','latent_severe_source_2024'),('2023–2025','three_year')]:
 p=lat.pivot(index=['iso3','group_id'],columns='scenario_id',values=col)
 z=np.log(p[positive]/p[negative]);d=z.rename('signed_log_burden_ratio').reset_index();d['window']=window;d['ratio']=np.exp(abs(z)).values
 d['stratum']=pd.cut(d.ratio,[1,1.5,2,4,np.inf],right=False,labels=['1.0–1.5','1.5–2.0','2.0–4.0','≥4.0']);directions.append(d)
d=pd.concat(directions);d.to_csv(OUT/'direction_pairs.csv',index=False)
records=[]
for window in d.window.unique():
 for country in ['ALL']+sorted(d.iso3.unique()):
  for st in ['1.0–1.5','1.5–2.0','2.0–4.0','≥4.0']:
   g=d[(d.window==window)&(d.stratum==st)];g=g if country=='ALL' else g[g.iso3==country]
   n=len(g);pos=int((g.signed_log_burden_ratio>0).sum());neg=int((g.signed_log_burden_ratio<0).sum())
   records.append(dict(window=window,iso3=country,stratum=st,n_pairs=n,positive=pos,negative=neg,zero=n-pos-neg,positive_fraction=pos/n if n else np.nan))
pd.DataFrame(records).to_csv(OUT/'direction_summary.csv',index=False)
f=pd.read_csv(FP);f=f[f.scenario_id.isin([positive,negative])]
params=f.groupby('iso3')[['calibrated_testing_log_sd','calibrated_severe_log_sd','calibrated_severe_nb_phi']].first()
for col in params:assert f.groupby('iso3')[col].nunique().max()==1
pred=pd.read_csv(HP);pred=pred[(pred.dataset=='process_anchored')&(pred.classifier=='shrinkage_linear_discriminant')&pred.package.isin(['testing','severe'])]
pred.groupby(['package','iso3','truth_active']).probability_active.agg(n='size',q025=lambda x:x.quantile(.025),median='median',q975=lambda x:x.quantile(.975)).reset_index().to_csv(OUT/'score_distributions.csv',index=False)
rows=[]
for iteration in range(100):
 seed=2026100900+iteration
 frame=regenerated_notifications(lat,seed)
 index=lat.set_index(['group_id','scenario_id']);aligned=index.loc[list(zip(frame.group_id,frame.scenario_id))]
 pars=params.loc[frame.iso3];n=len(frame)
 rng_t,rng_h,rng_b=[np.random.default_rng(x) for x in np.random.SeedSequence(seed+100000).spawn(3)]
 sd=pars.calibrated_severe_log_sd.to_numpy();phi=pars.calibrated_severe_nb_phi.to_numpy()
 base=rng_b.poisson(rng_b.gamma(phi,50/phi))
 for y in (2023,2024,2025):
  frame[f'feature_testing_log_index_{y}']=aligned[f'latent_testing_log_index_{y}'].to_numpy()+rng_t.normal(size=n)*pars.calibrated_testing_log_sd.to_numpy()
  mean=50*aligned[f'latent_severe_source_{y}'].to_numpy()/aligned.latent_severe_source_baseline.to_numpy()*np.exp(rng_h.normal(size=n)*sd-.5*sd**2)
  count=rng_h.poisson(rng_h.gamma(phi,mean/phi));frame[f'feature_severe_log_ratio_{y}']=np.log((count+.5)/(base+.5))
 result=dict(iteration=iteration,seed=seed)
 for name,streams in [('notifications',()),('index',('testing',)),('severe',('severe',)),('combined',('testing','severe'))]:
  p=leave_one_country_out_predictions(frame,feature_columns_for_package(frame,streams))
  result[name]=float(np.mean([roc_auc(g.truth_active,g.probability_active) for _,g in p.groupby('iso3')]))
 result['severe_minus_index']=result['severe']-result['index'];rows.append(result)
 if iteration%10==0:print(iteration,result,flush=True)
 pd.DataFrame(rows).to_csv(OUT/'noise_realisations.csv',index=False)
r=pd.DataFrame(rows);summary=[]
for col in ['notifications','index','severe','combined','severe_minus_index']:
 v=r[col];summary.append(dict(metric=col,n=len(v),mean=v.mean(),sd=v.std(),q025=v.quantile(.025),q975=v.quantile(.975),minimum=v.min(),maximum=v.max()))
pd.DataFrame(summary).to_csv(OUT/'noise_summary.csv',index=False)
(OUT/'manifest.json').write_text(json.dumps(dict(realisations=100,seeds=[2026100900,2026100999],packages=['notifications','index','severe','combined'],note='Fresh independent notification, index, severe and baseline noise; original conditional laws, frozen latent pairs and anchors; three replicates; LOCO refit per realisation. Age and serology not included.',hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [LP,FP,HP,Path(__file__)]}),indent=2))
print(pd.DataFrame(summary).to_string(index=False))
