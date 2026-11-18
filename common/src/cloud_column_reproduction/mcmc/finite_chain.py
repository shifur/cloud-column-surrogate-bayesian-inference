"""Finite-chain distance-floor constructions that preserve walker clusters."""
from __future__ import annotations
import numpy as np
from .energy_distance import replicated_subsample_ed,ReplicatedED

def geyer_ips_iat(series:np.ndarray)->float:
    """Deterministic initial-positive-sequence IAT for one transition summary."""
    x=np.asarray(series,dtype=np.float64)
    if x.ndim!=1 or len(x)<16 or not np.all(np.isfinite(x)): raise ValueError('IAT series must be finite 1-D with at least 16 transitions')
    x=x-np.mean(x);n=len(x);variance=float(np.dot(x,x)/n)
    if variance<=0: return float('inf')
    size=1<<(2*n-1).bit_length();f=np.fft.rfft(x,size);acov=np.fft.irfft(f*np.conjugate(f),size)[:n]/np.arange(n,0,-1);rho=acov/acov[0]
    total=0.0;previous=float('inf')
    for k in range(1,n-1,2):
        pair=float(rho[k]+rho[k+1])
        if pair<=0: break
        pair=min(pair,previous);previous=pair;total+=pair
    return max(1.0,1.0+2.0*total)

def select_mbb_block_length(single_ensemble:np.ndarray)->dict[str,object]:
    """Use transition-level walker means; walkers are never separated."""
    x=np.asarray(single_ensemble,dtype=np.float64)
    if x.ndim!=3 or x.shape[1]<2 or x.shape[2]!=11: raise ValueError('Expected (transition,walker,11)')
    taus=np.asarray([geyer_ips_iat(np.mean(x[:,:,j],axis=1)) for j in range(11)])
    if not np.all(np.isfinite(taus)): raise RuntimeError('Block rule cannot handle a stuck/constant transition summary')
    target=max(16.0,2.0*float(np.max(taus)));length=1<<int(np.ceil(np.log2(target)))
    if x.shape[0]//length<20: raise RuntimeError('Chain is too short for 20 moving-block-bootstrap blocks at selected dependence scale')
    return {'parameter_iat':taus,'worst_parameter_index':int(np.argmax(taus)),'worst_iat':float(np.max(taus)),
      'multiplier':2.0,'rounding':'next-power-of-two','minimum':16,'block_length':length,'transition_count':x.shape[0],
      'whole_walker_cluster_preserved':True}

def independent_ensemble_floor(normalized_chain:np.ndarray,**kwargs)->ReplicatedED:
    x=np.asarray(normalized_chain)
    if x.ndim!=4 or x.shape[0]!=2: raise ValueError("Independent-ensemble floor requires exactly two ensemble runs")
    return replicated_subsample_ed(x[0].reshape(-1,x.shape[-1]),x[1].reshape(-1,x.shape[-1]),**kwargs)

def moving_block_bootstrap_floor(single_ensemble:np.ndarray,*,block_length:int,replicates:int,seed:int,sample_size:int,ed_seeds:tuple[int,...],chunk_rows:int=512)->np.ndarray:
    x=np.asarray(single_ensemble,dtype=np.float64)
    if x.ndim!=3 or x.shape[-1]!=11 or block_length<1 or block_length>x.shape[0]: raise ValueError("Expected (transition,walker,11) and valid block length")
    rng=np.random.default_rng(seed); t=x.shape[0]; nblocks=int(np.ceil(t/block_length)); out=[]
    for _ in range(replicates):
        draws=[]
        for __ in range(2):
            starts=rng.integers(0,t-block_length+1,size=nblocks)
            draws.append(np.concatenate([x[s:s+block_length] for s in starts],axis=0)[:t].reshape(-1,11))
        out.append(replicated_subsample_ed(draws[0],draws[1],sample_size=sample_size,seeds=ed_seeds,chunk_rows=chunk_rows).mean)
    return np.asarray(out)
