"""Prior-normalized, memory-bounded posterior energy distances."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import t
from cloud_column_reproduction.domain import ParameterBox

def normalize_prior(x: np.ndarray, box: ParameterBox) -> np.ndarray:
    x = box.require_physical(x)
    return (x-np.asarray(box.lower))/(np.asarray(box.upper)-np.asarray(box.lower))

def denormalize_prior(z: np.ndarray, box: ParameterBox) -> np.ndarray:
    z=np.asarray(z,dtype=np.float64)
    if z.shape[-1]!=box.dimension or not np.all(np.isfinite(z)) or np.any(z < -32*np.finfo(float).eps) or np.any(z > 1+32*np.finfo(float).eps):
        raise ValueError("Normalized parameters must be finite and inside [0,1]")
    return np.asarray(box.lower)+np.clip(z,0,1)*(np.asarray(box.upper)-np.asarray(box.lower))

def _mean_distance(x: np.ndarray,y: np.ndarray,chunk_rows:int) -> float:
    total=0.0
    for start in range(0,len(x),chunk_rows): total += float(cdist(x[start:start+chunk_rows],y).sum(dtype=np.float64))
    return total/(len(x)*len(y))

def exact_energy_distance(x:np.ndarray,y:np.ndarray,*,chunk_rows:int=512)->float:
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    if x.ndim!=2 or y.ndim!=2 or x.shape[1]!=y.shape[1] or min(len(x),len(y))<1 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Energy-distance samples must be finite nonempty 2-D arrays with common dimension")
    if x.shape == y.shape and np.array_equal(x, y):
        return 0.0
    cross=_mean_distance(x,y,chunk_rows);within_x=_mean_distance(x,x,chunk_rows);within_y=_mean_distance(y,y,chunk_rows)
    squared=2*cross-within_x-within_y
    # The three pairwise sums can cancel at machine precision for identical
    # empirical measures presented in different row orders.
    scale=max(1.0,2*cross)
    if abs(squared) <= 64*np.finfo(np.float64).eps*scale: squared=0.0
    return float(np.sqrt(max(0.0,squared)))

@dataclass(frozen=True)
class ReplicatedED:
    estimates: np.ndarray
    mean: float
    standard_deviation: float
    standard_error: float
    ci95: tuple[float,float]
    sample_size_x: int
    sample_size_y: int
    seeds: tuple[int,...]

def replicated_subsample_ed(x:np.ndarray,y:np.ndarray,*,sample_size:int,seeds:tuple[int,...],chunk_rows:int=512,dimensions:tuple[int,...]|None=None)->ReplicatedED:
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    if dimensions is not None: x=x[:,dimensions];y=y[:,dimensions]
    if sample_size>len(x) or sample_size>len(y): raise ValueError("Predeclared ED subsample exceeds available posterior samples")
    values=[]
    for seed in seeds:
        rng=np.random.default_rng(seed)
        ix=rng.choice(len(x),sample_size,replace=False);iy=rng.choice(len(y),sample_size,replace=False)
        values.append(exact_energy_distance(x[ix],y[iy],chunk_rows=chunk_rows))
    a=np.asarray(values); sd=float(np.std(a,ddof=1)) if len(a)>1 else 0.0;se=sd/np.sqrt(len(a));crit=float(t.ppf(.975,len(a)-1)) if len(a)>1 else 0.0;mean=float(np.mean(a))
    return ReplicatedED(a,mean,sd,se,(max(0.,mean-crit*se),mean+crit*se),sample_size,sample_size,tuple(seeds))

def nested_subsample_indices(n_x:int,n_y:int,seeds:tuple[int,...])->dict[int,tuple[np.ndarray,np.ndarray]]:
    """One fixed permutation per replicate; every requested size uses its prefix."""
    out={}
    for seed in seeds:
        rng=np.random.default_rng(seed);out[seed]=(rng.permutation(n_x),rng.permutation(n_y))
    return out

def pairwise_ed_matrix(x:np.ndarray,y:np.ndarray,**kwargs)->tuple[np.ndarray,np.ndarray]:
    x=np.asarray(x);d=x.shape[1];means=np.empty((d,d));ses=np.empty((d,d))
    for i in range(d):
        for j in range(i,d):
            dims=(i,) if i==j else (i,j);r=replicated_subsample_ed(x,y,dimensions=dims,**kwargs)
            means[i,j]=means[j,i]=r.mean;ses[i,j]=ses[j,i]=r.standard_error
    return means,ses
