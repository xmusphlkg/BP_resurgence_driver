#!/usr/bin/env python3
"""Verify primary count generation and vary only the primary severe-outcome map."""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'script'),str(ROOT/'script/src'),str(Path(__file__).parent)]
import run_model2c_surveillance_design as runner
from pertussis_global_dynamics.bayes.model2c_recovery import negative_binomial_2_rng
from pertussis_global_dynamics.bayes.model2c_surveillance_design import leave_one_country_out_predictions,feature_columns_for_package
from run_ijid_revision_audits import LATENT, OUT, HOLDOUT, sha256, country_auc_table
from run_country_macro_revision_20260921 import macro_bootstrap
SOURCE=ROOT/'outputs/agent_measurement_model_upgrade_20260911/observation_features_calibrated.csv'
KEY=['group_id','scenario_id']
YEARS=(2023,2024,2025)

def main():
    latent=pd.read_csv(LATENT).set_index(KEY)
    features=pd.read_csv(SOURCE)
    features=features.loc[features.scenario_id.isin(['matched_transmission_truth','matched_reporting_truth'])].copy().reset_index(drop=True)
    features['truth_active']=features.scenario_id.eq('matched_transmission_truth').astype(int)
    severe_cols=[f'feature_severe_log_ratio_{y}' for y in YEARS]
    notification_cols=[f'feature_notifications_{y}_m{m:02d}' for y in YEARS for m in range(1,13)]
    specs=[]; max_error=0.; all_seeds=[]
    for _, row in features.iterrows():
        item=latent.loc[(row.group_id,row.scenario_id)].copy()
        item['group_id']=row.group_id; item['scenario_id']=row.scenario_id
        seed=runner._observation_seed(item,'main',20260726,int(row.observation_replicate))
        all_seeds.append(seed)
        rng=np.random.default_rng(np.random.SeedSequence(seed).spawn(6)[0])
        generated=[]
        for year in YEARS:
            for month in range(1,13):
                value=negative_binomial_2_rng(np.asarray([item[f'latent_notification_mean_{year}_m{month:02d}']]),max(float(item.nb_phi_month),.2),rng)[0]
                generated.append(np.log((value+.5)/(item[f'latent_notification_baseline_m{month:02d}']+.5)))
        max_error=max(max_error,float(np.max(abs(np.array(generated)-row[notification_cols].to_numpy(float)))))
        pair=latent.loc[row.group_id]
        signals=np.log(pair[[f'latent_severe_source_{y}' for y in YEARS]].to_numpy(float)/pair.latent_severe_source_baseline.to_numpy(float)[:,None])
        native=np.log(item[[f'latent_severe_source_{y}' for y in YEARS]].to_numpy(float)/item.latent_severe_source_baseline)
        specs.append((seed, signals.mean(axis=0),native,float(row.calibrated_severe_log_sd),float(row.calibrated_severe_nb_phi)))
    assert max_error < 1e-12 and len(set(all_seeds))==len(all_seeds)
    records=[]; country_rows=[]; primary_error=None
    for factor in (0.,.25,.5,.75,1.):
        for drift_multiplier in (1.,2.):
            frame=features.copy(); generated=[]
            for seed, midpoint,native,sd,phi in specs:
                children=np.random.SeedSequence(seed+99173).spawn(4)
                rng=np.random.default_rng(children[1]); baseline_rng=np.random.default_rng(children[3])
                baseline=negative_binomial_2_rng(np.array([50.]),phi,baseline_rng)[0]
                values=[]
                for signal in midpoint+factor*(native-midpoint):
                    drift_sd=sd*drift_multiplier
                    drift=np.exp(rng.normal(-.5*drift_sd**2,drift_sd))
                    count=negative_binomial_2_rng(np.array([50*np.exp(signal)*drift]),phi,rng)[0]
                    values.append(np.log((count+.5)/(baseline+.5)))
                generated.append(values)
            frame[severe_cols]=np.array(generated)
            if factor==1 and drift_multiplier==1:
                primary_error=float(np.max(abs(frame[severe_cols].to_numpy()-features[severe_cols].to_numpy())))
                assert primary_error<1e-12,primary_error
            pred=leave_one_country_out_predictions(frame,feature_columns_for_package(frame,('severe',)))
            pred['probability_active']=pred.probability_active.clip(1e-9,1-1e-9)
            country,boot=macro_bootstrap(pred)
            lo,hi=np.quantile(boot,[.025,.975])
            records.append(dict(severe_contrast_factor=factor,drift_multiplier=drift_multiplier,annual_drift_sd=specs[0][3]*drift_multiplier,country_macro_auc=country.mean(),lower_95=lo,upper_95=hi))
            country_rows.extend(dict(severe_contrast_factor=factor,drift_multiplier=drift_multiplier,iso3=k,auc=v) for k,v in country.items())
            print(records[-1],flush=True)
    result=pd.DataFrame(records)
    result.to_csv(OUT/'primary_severe_mapping_sensitivity.csv',index=False)
    result.to_csv(ROOT/'outputs/derived_tables/primary_severe_mapping_sensitivity.csv',index=False)
    pd.DataFrame(country_rows).to_csv(OUT/'primary_severe_mapping_country_auc.csv',index=False)
    frozen=pd.read_csv(HOLDOUT)
    frozen=frozen.loc[frozen.dataset.eq('process_anchored') & frozen.package.eq('notifications')]
    replay=leave_one_country_out_predictions(features,notification_cols)
    check=replay.merge(frozen,on=KEY+['observation_replicate'],suffixes=('_replay','_frozen'),validate='one_to_one')
    score_error=float(abs(check.probability_active_replay-check.probability_active_frozen).max())
    assert score_error<1e-12
    contract=dict(notification_max_feature_reproduction_error=max_error,notification_max_score_reproduction_error=score_error,unique_seed_count=len(set(all_seeds)),observation_rows=len(all_seeds),severe_native_max_feature_reproduction_error=primary_error,
        null_contract='Same NB2 law, phi floor 0.2, fixed pair-shared baselines and +0.5 ratio. Null vectorises independent gamma-Poisson draws under fresh seeds; primary uses hashed row seeds and separate stream children. Random-number ordering differs; distribution does not.',
        preprocessing='Full original group folds and country holdouts; standardisation fitted on training data only. No label-specific baseline or dispersion. Labels select class membership and independent primary random streams.',
        mapping='Scale paired log burden-to-baseline contrasts about their pair midpoint before original NB2 counts and drift. Same primary seeds, observed baseline, dispersion and three replicates. Refit same LOCO rule for each map; country-first/pair-second bootstrap conditional on predictions.',
        input_hashes={str(p.relative_to(ROOT)):sha256(p) for p in (SOURCE,LATENT,HOLDOUT)},script_sha256=sha256(Path(__file__)))
    (OUT/'primary_generation_and_mapping_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
if __name__=='__main__': main()
