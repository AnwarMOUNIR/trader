"""Acquire and score Model 16 on its preregistered unseen-stock lockbox."""
from __future__ import annotations
import hashlib, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
from market_research.acquire import AlpacaAcquisitionClient, acquire_symbol_year
from market_research.model15_acquire_mid import _resample_30min
from market_research.model15_research import FEATURES_NEW, REPORT_DIR, _portfolio
import market_research.model15_research as research
from market_research.trend_experiment import FEATURES, make_rows
from market_research.validation import require_unscored, verify_manifest

ROOT=Path(__file__).resolve().parents[2]
REPORT_DIR=ROOT/'reports/model16'
PROTO=REPORT_DIR/'protocol.json'; LOCK=REPORT_DIR/'lockbox_protocol.json'
DATA=ROOT/'data/challenges/model16/lockbox'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def verify_frozen():
    p=json.loads(PROTO.read_text()); l=json.loads(LOCK.read_text())
    require_unscored(l)
    if sha(ROOT/p['artifact'])!=p['sha256'] or p['sha256']!=l['model_sha256']:
        raise RuntimeError('Frozen model checksum mismatch')
    if p['status']!='frozen_research_candidate_lockbox_pending':
        raise RuntimeError('Model protocol is not awaiting first lockbox')
    trained=set(p['training_symbols']); excluded=set(p.get('excluded_evaluation_symbols',[]))
    overlap={x['ticker'] for x in l['symbols']} & (trained|excluded|set(l['training_tickers_excluded']))
    if overlap: raise RuntimeError(f'Lockbox overlap: {sorted(overlap)}')
    return p,l

def _acquire_ytd(client, symbol, year=2026):
    now=datetime.now(timezone.utc)
    target=DATA/f'raw/feed=sip/timeframe=1Min/symbol={symbol}/{symbol}_{year}_1min.parquet'
    if target.exists():
        return {'status':'COMPLETED_CACHED','bars_count':len(pd.read_parquet(target,columns=['timestamp_utc'])),'file_path':str(target)}
    end_day=now.date() if now.hour>=21 else (now.date()-pd.Timedelta(days=1)).date()
    end=end_day.isoformat()+'T21:00:00Z'  # regular-session data only; excludes future/current after-hours requests
    bars=client.fetch_bars_chunk(symbol,f'{year}-01-01T00:00:00Z',end,timeframe='1Min',adjustment='raw')
    if bars:
        df=pd.DataFrame(bars).rename(columns={'t':'timestamp_utc','o':'open','h':'high','l':'low','c':'close','v':'volume','n':'trade_count','vw':'vwap'})
        df['symbol']=symbol; df['feed']='sip'; df['source']='alpaca'; df['adjustment']='raw'; target.parent.mkdir(parents=True,exist_ok=True)
        tmp=target.with_suffix('.parquet.tmp'); df.to_parquet(tmp,index=False,compression='snappy'); tmp.replace(target)
    return {'status':'COMPLETED' if bars else 'EMPTY','bars_count':len(bars),'file_path':str(target) if bars else None,'request_end_utc':end}

def acquire():
    p,l=verify_frozen(); symbols=[x['ticker'] for x in l['symbols']]+['SPY']
    years=[2024,2025,2026]
    # SPY 2024 already exists from the prior audited archive; only acquire if missing.
    jobs=[(s,y) for s in symbols for y in years if y<2026 and not (s=='SPY' and y==2024 and (ROOT/f'data/raw/feed=sip/timeframe=1Min/symbol=SPY/SPY_{y}_1min.parquet').exists())]
    jobs += [(s,2026) for s in symbols if s!='SPY' or not (DATA/'raw/feed=sip/timeframe=1Min/symbol=SPY/SPY_2026_1min.parquet').exists()]
    results=[]
    with AlpacaAcquisitionClient(feed='sip') as client:
        with ThreadPoolExecutor(max_workers=4) as pool:
            fs={pool.submit(_acquire_ytd,client,s,y) if y==2026 else pool.submit(acquire_symbol_year,client,s,y,output_base=DATA/'raw'): (s,y) for s,y in jobs}
            for f in as_completed(fs):
                s,y=fs[f]
                try: r=f.result(); results.append({'symbol':s,'year':y,**r}); print(f"{s} {y}: {r['status']} {r['bars_count']}",flush=True)
                except Exception as e: results.append({'symbol':s,'year':y,'status':'FAILED','error':repr(e)}); print(f'{s} {y}: FAILED {e!r}',flush=True)
    # Resample every available year into a lockbox-only location. Existing SPY 2024 is read from audited raw archive.
    files=[]
    acquired_by_key={(item.get('symbol'), item.get('year')): item for item in results}
    for s in symbols:
        parts=[]
        for y in years:
            path=DATA/f'raw/feed=sip/timeframe=1Min/symbol={s}/{s}_{y}_1min.parquet'
            if not path.exists():
                cached=acquired_by_key.get((s,y),{}).get('file_path')
                if cached and Path(cached).exists(): path=Path(cached)
            if s=='SPY' and y==2024 and not path.exists(): path=ROOT/f'data/raw/feed=sip/timeframe=1Min/symbol=SPY/SPY_{y}_1min.parquet'
            if path.exists(): parts.append(pd.read_parquet(path))
        if not parts: continue
        bars=_resample_30min(pd.concat(parts,ignore_index=True),s)
        target=DATA/'candles/30min'/f'{s}_30min.parquet'; target.parent.mkdir(parents=True,exist_ok=True)
        bars.to_parquet(target,index=False,compression='snappy')
        files.append({'symbol':s,'rows':len(bars),'sha256':sha(target),'path':str(target.relative_to(ROOT))})
        print(f'Resampled {s}: {len(bars)} bars',flush=True)
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'lockbox_symbols':[x['ticker'] for x in l['symbols']], 'years':years,'acquisition_results':results,'resampled':files}
    (REPORT_DIR/'lockbox_data_manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest

def build_features(symbols):
    root=DATA/'candles/30min'; spy=pd.read_parquet(root/'SPY_30min.parquet').sort_values('timestamp_utc').copy()
    spy['timestamp_utc']=pd.to_datetime(spy.timestamp_utc,utc=True); g=spy.groupby('session_date',sort=False)
    spy['spy_r1']=g.close.pct_change(1,fill_method=None); spy['spy_r3']=g.close.pct_change(3,fill_method=None); spy['spy_range1']=(spy.high-spy.low)/spy.open
    spy['spy_entry_open']=g.open.shift(-1); spy['spy_exit_close']=g.close.shift(-3); spy['spy_exec_return']=spy.spy_exit_close/spy.spy_entry_open-1
    market=spy[['timestamp_utc','spy_r1','spy_r3','spy_range1','spy_entry_open','spy_exec_return']]
    sm=market.set_index('timestamp_utc'); frames=[]; bars={}
    for s in symbols:
        path=root/f'{s}_30min.parquet'; raw=pd.read_parquet(path).sort_values('timestamp_utc').copy(); raw['timestamp_utc']=pd.to_datetime(raw.timestamp_utc,utc=True)
        g=raw.groupby('session_date',sort=False); raw['entry_open']=g.open.shift(-1); raw['exit_close']=g.close.shift(-3); raw['entry_time']=g.timestamp_utc.shift(-1); raw['exit_bar_time']=g.timestamp_utc.shift(-3); raw['exec_return']=raw.exit_close/raw.entry_open-1
        r1=g.close.pct_change(1,fill_method=None); rv=r1.rolling(65,min_periods=20).std()
        spy_r1=raw.timestamp_utc.map(sm.spy_r1); spy_r3=raw.timestamp_utc.map(sm.spy_r3); spy_future=raw.timestamp_utc.map(sm.spy_exec_return)
        cov=r1.rolling(65,min_periods=30).cov(spy_r1); var=spy_r1.rolling(65,min_periods=30).var(); beta=(cov/var.replace(0,np.nan)).clip(-3,3).fillna(1)
        resid=r1-beta*spy_r1; residvol=resid.rolling(65,min_periods=30).std().fillna(rv)
        bars[s]=raw.set_index('timestamp_utc',drop=False); feats=make_rows(path,market=market,deadzone=0.0)
        extra=raw[['timestamp_utc','entry_open','exit_close','entry_time','exit_bar_time','exec_return']].copy(); extra['rv_30m']=rv.to_numpy(); extra['rv_resid_30m']=residvol.to_numpy(); extra['beta_spy']=beta.to_numpy(); extra['spy_exec_return']=spy_future.to_numpy(); extra['spy_entry_open']=raw.timestamp_utc.map(sm.spy_entry_open).to_numpy(); extra['resid_r1']=resid.to_numpy()
        feat_idx=feats.set_index('timestamp_utc'); extra['resid_r3']=feat_idx.r3.reindex(extra.timestamp_utc).to_numpy()-beta.to_numpy()*spy_r3.to_numpy(); feats=feats.merge(extra,on='timestamp_utc',how='inner',validate='one_to_one')
        den=feats.rv_30m.clip(lower=1e-5); feats['r1_volnorm']=feats.r1/den; feats['r3_volnorm']=feats.r3/(den*np.sqrt(3)); feats['r6_volnorm']=feats.r6/(den*np.sqrt(6)); feats['range_volnorm']=feats.range1/den; feats['body_volnorm']=feats.body1/den; feats['vwap_volnorm']=feats.vwap_gap/den; feats['regime_post']=1.0; feats['regime_mid']=0.0
        good=feats[[f for f in FEATURES_NEW if not f.startswith('cs_')]].replace([np.inf,-np.inf],np.nan).notna().all(axis=1); good &= feats.exec_return.notna()&feats.entry_open.gt(0)&feats.exit_close.gt(0); feats=feats.loc[good].copy(); feats['symbol']=s; frames.append(feats)
    d=pd.concat(frames,ignore_index=True)
    for source,dest in [('r1_volnorm','cs_r1_rank'),('r3_volnorm','cs_r3_rank'),('vwap_volnorm','cs_vwap_rank')]: d[dest]=d.groupby('timestamp_utc',sort=False)[source].rank(pct=True)-.5
    d=d.loc[d[FEATURES_NEW].replace([np.inf,-np.inf],np.nan).notna().all(axis=1)].copy(); bars['SPY']=spy.set_index('timestamp_utc',drop=False)
    d=d.loc[(d.timestamp_utc>='2025-01-02')&(d.timestamp_utc<'2026-10-29')].copy()
    return d,bars

def score():
    p,l=verify_frozen(); manifest_path=REPORT_DIR/'lockbox_data_manifest.json'
    if not manifest_path.exists(): raise RuntimeError('Acquire lockbox data before scoring')
    manifest=json.loads(manifest_path.read_text()); missing=[x['ticker'] for x in l['symbols'] if not (DATA/'candles/30min'/f"{x['ticker']}_30min.parquet").exists()]
    if missing: raise RuntimeError(f'No replacement allowed: registered stocks unavailable: {missing}')
    verify_manifest(ROOT, manifest, [x['ticker'] for x in l['symbols']], DATA/'candles/30min')
    payload=load(ROOT/p['artifact']); d,bars=build_features([x['ticker'] for x in l['symbols']]); raw=payload['estimator'].predict(d[payload['features']]); d['raw_expected_return']=payload['calibration_intercept']+payload['calibration_slope']*raw
    # Frozen long-only control: trade only expected returns at or above the registered 20 bps floor.
    d['expected_return']=np.where(d.raw_expected_return.ge(0.002),d.raw_expected_return,0.0)
    cost_results={}
    for bps in (10,15,25):
        research.RT_COST_BPS=float(bps)
        cost_results[str(bps)]=_portfolio(d,d.expected_return.to_numpy(),bars,f'model16_long_only_20bps_{bps}bps',market_neutral=False)
    research.RT_COST_BPS=10.0
    base=cost_results['10']
    preds=d[['symbol','timestamp_utc','entry_time','exit_bar_time','raw_expected_return','expected_return','exec_return','rv_30m','beta_spy']].copy()
    preds.to_parquet(REPORT_DIR/'lockbox_predictions.parquet',index=False,compression='snappy')
    result={'status':'lockbox_scored_frozen_candidate','model_sha256':p['sha256'],'registered_symbols':[x['ticker'] for x in l['symbols']],'rows':int(len(d)),'date_min':str(d.timestamp_utc.min()),'date_max':str(d.timestamp_utc.max()),'portfolio_by_round_trip_cost_bps':cost_results,'candidate_diagnostics':{'mean_abs_raw_forecast_bps':float(np.abs(d.raw_expected_return).mean()*10000),'eligible_long_signals':int(d.expected_return.gt(0).sum()),'eligible_signal_rate':float(d.expected_return.gt(0).mean()),'policy':'long-only; raw calibrated expected return >= 20 bps'}}
    (REPORT_DIR/'lockbox_results.json').write_text(json.dumps(result,indent=2)); l['status']='scored'; l['scored_utc']=datetime.now(timezone.utc).isoformat(); l['results']='reports/model16/lockbox_results.json'; LOCK.write_text(json.dumps(l,indent=2)); print(json.dumps(result,indent=2)); return result

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser(); a.add_argument('action',choices=['acquire','score','all']); args=a.parse_args()
    if args.action in ('acquire','all'): acquire()
    if args.action in ('score','all'): score()
