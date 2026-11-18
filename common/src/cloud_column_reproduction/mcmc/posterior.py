"""Posterior representation, deterministic summaries, and plot source data."""
from __future__ import annotations
import numpy as np

def require_chain4d(x:np.ndarray,*,ensembles:int|None=None)->np.ndarray:
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=4 or x.shape[-1]!=11 or not np.all(np.isfinite(x)): raise ValueError("Expected finite (ensemble,transition,walker,11) retained chain")
    if ensembles is not None and x.shape[0]!=ensembles: raise ValueError("Independent ensemble count mismatch")
    return x

def pool_retained(x:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    x=require_chain4d(x); labels=np.broadcast_to(np.arange(x.shape[0])[:,None,None],x.shape[:3]).reshape(-1)
    return x.reshape(-1,11),labels

def histogram_sources(chains:dict[str,np.ndarray],lower:np.ndarray,upper:np.ndarray,*,bins:int=80)->dict[str,np.ndarray]:
    edges=np.stack([np.linspace(lower[j],upper[j],bins+1) for j in range(11)])
    out={"edges":edges}
    for method,chain in chains.items():
        pooled,_=pool_retained(chain);out[method]=np.stack([np.histogram(pooled[:,j],bins=edges[j])[0]/len(pooled) for j in range(11)])
    return out

def ensemble_checks(x:np.ndarray,lower:np.ndarray,upper:np.ndarray)->dict[str,np.ndarray]:
    x=require_chain4d(x,ensembles=2); a=x[0].reshape(-1,11);b=x[1].reshape(-1,11)
    q=np.array([.05,.25,.5,.75,.95]); qa=np.quantile(a,q,axis=0);qb=np.quantile(b,q,axis=0)
    ecdf=np.empty(11)
    for j in range(11):
        grid=np.sort(np.r_[a[:,j],b[:,j]]);ecdf[j]=np.max(np.abs(np.searchsorted(np.sort(a[:,j]),grid,side='right')/len(a)-np.searchsorted(np.sort(b[:,j]),grid,side='right')/len(b)))
    width=upper-lower
    boundary=np.stack([np.mean((v<=lower+.01*width)|(v>=upper-.01*width),axis=0) for v in (a,b)])
    edges=np.linspace(0,1,21); za=(a-lower)/width;zb=(b-lower)/width
    occupancy=np.stack([[np.histogram(v[:,j],edges)[0]/len(v) for j in range(11)] for v in (za,zb)])
    selected=[]
    for v in (za,zb): selected.append(np.stack([np.histogram2d(v[:,i],v[:,j],bins=(edges,edges))[0]/len(v) for i,j in ((0,1),(2,3))]))
    return {"quantile_probabilities":q,"ensemble_quantiles":np.stack([qa,qb]),"maximum_ecdf_difference":ecdf,
      "boundary_occupancy":boundary,"mode_occupancy_bin_edges_normalized":edges,
      "marginal_mode_occupancy":occupancy,"selected_joint_mode_occupancy":np.stack(selected),
      "selected_joint_pair_indices":np.asarray(((0,1),(2,3)))}
