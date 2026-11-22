"""Nonlinear OCP adapter for the warm-start branch's existing flight runtime.

The runtime, guidance and PX4 trajectory tracker retain their original policy.
Only the state, input and within-stage interpolation change with this model.
"""
from __future__ import annotations
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
import time
import numpy as np
from px4_nmpc_core.paths import repo_root
from px4_nmpc_controller.px4_nmpc_ipopt.solver import SolveResult
from px4_nmpc_core.math_utils import quat_wxyz_to_rotation,euler_to_quat_wxyz


def extension():
    directory=repo_root()/'build/nonlinear_slalom_v2'
    if str(directory) not in sys.path:sys.path.insert(0,str(directory))
    import _slalom
    return _slalom


class NonlinearTrajectoryModel:
    name='nonlinear_thrust_body_rate';nx=9;nu=4
    has_yaw=True;nonlinear_trajectory=True
    state_names=('pN','pE','pD','vN','vE','vD','roll','pitch','yaw')
    input_names=('thrust_N','omega_x','omega_y','omega_z')
    def __init__(self,p,interface='body_rate'):
        self.p=p;self.native=extension();self.g=p.gravity;self.dt=p.dt;self.drag=np.zeros(3)
        self.interface=interface
    def state_from_measurement(self,pos,vel,rpy,accel=None,**kwargs):return np.r_[pos,vel,rpy]
    def clip_state(self,x):return np.array(x,float,copy=True)
    def hover_input(self,d):return np.array([self.p.mass*self.p.gravity,0.,0.,0.])
    def command_thrust_accel(self,force_newton):return float(force_newton)/self.p.mass
    def step(self,x,u,d,yaw_ref,h):
        # Independent Params avoids changing the solver's sampling interval.
        p=self.native.Params();p.mass=self.p.mass;p.gravity=self.p.gravity;p.dt=float(h)
        return self.native.step(np.asarray(x),np.asarray(u),p)
    def state_at(self,plan,t):
        if t>plan.t_end:return plan.X[:,-1].copy()
        k,tau=plan.stage(t)
        if t<plan.t0:tau=0.
        return self.step(plan.X[:,k],plan.U[:,k],None,0.,tau) if tau>0 else plan.X[:,k].copy()
    def trajectory_at(self,plan,t):
        x=self.state_at(plan,t)
        if t>plan.t_end:return x[:3],np.zeros(3),np.zeros(3)
        k,_=plan.stage(t);a=self.native.rhs(x,plan.U[:,k],self.p)[3:6]
        return x[:3],x[3:6],a
    def yaw_at(self,plan,t):
        x=self.state_at(plan,t)
        if t>plan.t_end:return float(x[8]),0.
        k,_=plan.stage(t);return float(x[8]),float(self.native.rhs(x,plan.U[:,k],self.p)[8])


class NonlinearOcpSolver:
    _PARAMETER_FIELDS=('N','dt','mass','gravity','vmax','tilt','tmin','tmax',
        'rate_xy','rate_z','vehicle_radius','margin','heading_epsilon','yaw_heading_enabled','yaw_reference','lower',
        'upper','qp','qv','qpt','qvt','watt','wyaw','wt','wrate','wslack','wdu','obstacles')
    def __init__(self,cfg):
        start=time.perf_counter();self.cfg=cfg;self.native=extension();self.interface=cfg['interface']
        if self.interface not in ('trajectory','body_rate'):
            raise ValueError('Nonlinear prediction supports trajectory or direct body-rate control')
        self.backend=cfg['backend'];self.backend_error='';self.N=int(cfg['N']);self.dt=float(cfg['dt']);self.K=int(cfg['max_obstacles'])
        if self.K!=10:raise ValueError('The ported native nonlinear formulation currently requires 10 obstacle slots')
        self.log=open(os.environ['PX4_NMPC_NONLINEAR_LOG'],'a',buffering=1) if os.environ.get('PX4_NMPC_NONLINEAR_LOG') else None
        self.trace_log=open(os.environ['PX4_NMPC_NONLINEAR_TRACE'],'a',buffering=1) if os.environ.get('PX4_NMPC_NONLINEAR_TRACE') else None
        p=self.native.Params();p.N=self.N;p.dt=self.dt;p.mass=float(cfg.get('nonlinear_mass',2.0643076923))
        p.gravity=float(cfg['model']['g']);p.vmax=float(cfg['model']['vmax']);p.tilt=float(cfg['model']['tilt_max'])
        # Obstacles arrive already inflated by the unchanged runtime. Do not
        # add the vehicle radius or safety margin for a second time.
        self.tracking_margin=max(0.,float(cfg.get('nonlinear_tracking_margin',0.)))
        self.margin_adaptive=bool(cfg.get('nonlinear_margin_adaptive',True))
        self.recovery_buffer=max(0.,float(cfg.get('nonlinear_recovery_buffer',.02)))
        p.vehicle_radius=0.;p.margin=self.tracking_margin;p.obstacles=np.zeros((self.K,4))
        w=cfg['weights'];p.qp=np.asarray(w['pos']);p.qv=np.asarray(w['vel']);p.qpt=np.asarray(w['pos_terminal']);p.qvt=np.asarray(w['vel_terminal'])
        p.wslack=float(w['slack_obstacle_l2'])
        self.profile=cfg.get('nonlinear_profile','matched')
        if self.profile in ('matched','matched_fixed_yaw'):
            # Match the old acceleration envelope and near-hover cost units.
            # a_xy ~= g*[pitch,roll], j_xy ~= g*[q,p]. No guidance change.
            p.tmin=p.mass*(p.gravity-float(cfg['model']['acc_down_max']))
            p.tmax=p.mass*(p.gravity+float(cfg['model']['acc_up_max']))
            p.rate_xy=float(np.min(cfg['model']['jerk_max'][:2]))/p.gravity
            p.rate_z=.8
            p.watt=float(w['acc'])*p.gravity**2
            p.wt=float(w['acc'])/p.mass**2
            p.wrate=float(w['jerk'])*p.gravity**2
            p.wdu=np.array([float(w['jerk'])/(p.mass**2*p.dt**2),.05,.05,.05])
            p.heading_epsilon=.8
            p.yaw_heading_enabled=self.profile!='matched_fixed_yaw'
        elif self.profile!='ported':raise ValueError('nonlinear_profile must be ported, matched, or matched_fixed_yaw')
        rate_slew=float(cfg.get('nonlinear_rate_slew_weight',-1.))
        if not np.isfinite(rate_slew):raise ValueError('nonlinear_rate_slew_weight must be finite')
        if rate_slew>=0.:
            wdu=np.asarray(p.wdu).copy()
            wdu[1:4]=rate_slew
            p.wdu=wdu
        self.p=p;self.model=NonlinearTrajectoryModel(p,self.interface);self.nx=9;self.nu=4
        self.nz=14*self.N;self.ng=(10+16+self.K)*self.N;self.build_s=time.perf_counter()-start
        self.budget=int(cfg['max_iter']);self.tol=float(cfg['tol']);self.backtracking=float(cfg.get('nonlinear_backtracking_min',.03));self.previous=None
        from px4_nmpc_controller.px4_nmpc_cg2o.config import load_cg2o_config
        from px4_nmpc_controller.px4_nmpc_cg2o.solver import deep_merge
        settings=deep_merge(load_cg2o_config(cfg.get('cg2o_config_path')),cfg.get('cg2o_overrides') or {})
        self.warm_strategy=int(settings.get('warm_start',{}).get('strategy',1))
        self.ipopt_warm_mode=str((cfg.get('ipopt_cpp_overrides') or {}).get('warm_start','primal_dual'))
        self.session=self.native.CGSession() if self.backend=='cg2o' else self.native.IPOPTSession()
        settings.setdefault('ispd',{})['ineq_backtracking_step_min']=self.backtracking
        self.settings=settings
        if self.backend=='cg2o':self.session.configure(settings)
        if self.log:
            fields='N dt mass gravity vmax tilt tmin tmax rate_xy rate_z vehicle_radius margin heading_epsilon yaw_heading_enabled yaw_reference lower upper qp qv qpt qvt watt wyaw wt wrate wslack wdu'.split()
            metadata=dict(profile=self.profile,parameters={k:getattr(p,k).tolist() if isinstance(getattr(p,k),np.ndarray) else getattr(p,k) for k in fields},cg2o_settings=settings,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),native_sha256=hashlib.sha256(Path(self.native.__file__).read_bytes()).hexdigest())
            Path(self.log.name).with_suffix('.parameters.json').write_text(json.dumps(metadata,indent=2))
        self.rows=[]
    def reset_warm_start(self):
        self.previous=None
        self._reset_session()
    def _reset_session(self):
        if self.backend=='cg2o':
            self.session=self.native.CGSession();self.session.configure(self.settings)
        else:self.session.reset()
    @staticmethod
    def pack(X,U,S=None):
        slack=np.zeros(U.shape[1]) if S is None else np.asarray(S,float).reshape(-1)
        return np.r_[X[:,1:].T.ravel(),U.T.ravel(),slack]
    @staticmethod
    def _hash_value(digest,name,value):
        array=np.ascontiguousarray(value,dtype='<f8');encoded=name.encode()
        digest.update(struct.pack('<I',len(encoded)));digest.update(encoded)
        digest.update(struct.pack('<I',array.ndim))
        if array.ndim:digest.update(struct.pack('<'+'Q'*array.ndim,*array.shape))
        digest.update(array.tobytes())
    def problem_sha256(self,b):
        """Hash every numerical datum seen by both native NLP backends."""
        digest=hashlib.sha256(b'nonlinear-direct-rate-problem-v1')
        for name in self._PARAMETER_FIELDS:self._hash_value(digest,'p.'+name,getattr(b.p,name))
        for name in ('x0','reference','uprev'):self._hash_value(digest,name,getattr(b,name))
        return digest.hexdigest()
    def initial_sha256(self,z):
        digest=hashlib.sha256(b'nonlinear-direct-rate-initial-v1')
        self._hash_value(digest,'z0',z)
        return digest.hexdigest()
    def cold(self,b):
        X=np.zeros((9,self.N+1));U=np.zeros((4,self.N));X[:,0]=b.x0;p=b.p
        for k in range(self.N):
            x=X[:,k];vdes=b.reference[3:,k+1]+.5*(b.reference[:3,k+1]-x[:3]);vdes*=min(1.,p.vmax/max(1e-12,np.linalg.norm(vdes)))
            a=np.clip(2.*(vdes-x[3:6]),-2.,2.);yaw=x[8];an=np.cos(yaw)*a[0]+np.sin(yaw)*a[1];ae=-np.sin(yaw)*a[0]+np.cos(yaw)*a[1]
            force=np.linalg.norm(a-[0,0,p.gravity]);target=np.array([np.arcsin(np.clip(ae/force,-.4,.4)),np.arctan2(-an,p.gravity-a[2]),0.])
            edot=3*(target-x[6:]);phi,theta=x[6:8]
            rates=[edot[0]-np.sin(theta)*edot[2],np.cos(phi)*edot[1]+np.sin(phi)*np.cos(theta)*edot[2],-np.sin(phi)*edot[1]+np.cos(phi)*np.cos(theta)*edot[2]]
            U[:,k]=np.r_[np.clip(p.mass*force,p.tmin,p.tmax),np.clip(rates,[-p.rate_xy,-p.rate_xy,-p.rate_z],[p.rate_xy,p.rate_xy,p.rate_z])]
            X[:,k+1]=self.native.step(x,U[:,k],p)
        return self.pack(X,U)
    def solve_ocp(self,x0,pref,vref,obstacles,d=None,uprev=None,yaw_ref=0.,arena_lb=None,arena_ub=None,shift=None,**kwargs):
        b=self.native.Problem();b.p=self.p;b.x0=np.asarray(x0,float);b.reference=np.vstack((pref,vref));b.uprev=self.model.hover_input(d) if uprev is None else np.asarray(uprev,float)
        b.p.yaw_reference=float(yaw_ref)
        obs=np.asarray(obstacles,float).reshape(-1,4);padding=np.tile([1000.,1000.,1000.,.1],(self.K,1));padding[:len(obs)]=obs;b.p.obstacles=padding
        base_clearance=float(np.min(np.linalg.norm(padding[:,:3]-b.x0[:3],axis=1)-padding[:,3]))
        b.p.margin=(min(self.tracking_margin,max(0.,base_clearance-self.recovery_buffer))
                    if self.margin_adaptive else self.tracking_margin)
        if arena_lb is not None:b.p.lower=np.asarray(arena_lb,float)
        if arena_ub is not None:b.p.upper=np.asarray(arena_ub,float)
        b.build()
        start_init=time.perf_counter()
        warm=self.previous is not None and shift is not None
        if warm:
            X,U,S=b.states(self.previous),b.inputs(self.previous),b.slacks(self.previous);n=max(0,min(self.N-1,int(shift)))
            for _ in range(n):
                X=np.column_stack((b.x0,X[:,2:],self.native.step(X[:,-1],U[:,-1],b.p)));U=np.column_stack((U[:,1:],U[:,-1]));S=np.r_[S[1:],S[-1]]
            # PX4 may not lie exactly on the old plan.  Retain the shifted
            # control homotopy, but rebuild every state with the shared RK4
            # dynamics from the current measurement.  Merely replacing x0
            # leaves a large first dynamics defect and caused repeated SITL
            # warm-start failures after tracking disturbances.
            X[:,0]=b.x0
            for k in range(self.N):X[:,k+1]=self.native.step(X[:,k],U[:,k],b.p)
            z0=self.pack(X,U,S)
        else:z0=self.cold(b)
        initialization_ms=(time.perf_counter()-start_init)*1000
        problem_sha256=self.problem_sha256(b);initial_sha256=self.initial_sha256(z0)
        if self.trace_log:
            self.trace_log.write(json.dumps(dict(problem_index=len(self.rows),
                x0=b.x0.tolist(),pref=b.reference[:3].tolist(),vref=b.reference[3:].tolist(),
                obstacles=obs.tolist(),uprev=b.uprev.tolist(),yaw_ref=float(yaw_ref),
                arena_lb=b.p.lower.tolist(),arena_ub=b.p.upper.tolist(),
                shift=-1 if shift is None else int(shift),
                problem_sha256=problem_sha256,initial_sha256=initial_sha256))+'\n')
        initial=b.evaluate(z0);initial_states=b.states(z0)[:3,1:].T
        distances=np.linalg.norm(initial_states[:,None,:]-padding[None,:,:3],axis=2)
        penetrations=padding[None,:,3]-distances
        initial_obstacle_rows_violated=int(np.count_nonzero(penetrations>0.))
        initial_max_obstacle_penetration=float(max(0.,np.max(penetrations)))
        begin=time.perf_counter();solver='cg2o' if self.backend=='cg2o' else 'ipopt'
        if self.backend=='cg2o':
            result=self.session.solve(b,z0,self.budget,self.tol,self.backtracking,int(shift) if warm else -1,self.warm_strategy)
        else:
            result=self.session.solve(b,z0,self.budget,self.tol,int(shift) if warm else -1,self.ipopt_warm_mode=='primal_dual')
        native_call_ms=(time.perf_counter()-begin)*1000
        wall=initialization_ms+native_call_ms;z=result.pop('z')
        usable=bool(np.isfinite(z).all() and result['final_max_eq_residual']<=self.tol and result['final_max_ineq_violation']<=self.tol)
        if usable:self.previous=z.copy()
        else:
            # Preserve the last accepted primal plan.  A failed receding-horizon
            # update must not force the next nonlinear solve back to the
            # obstacle-ignorant cold rollout (which can also switch homotopy).
            # Failed dual/slack state is unsafe to reuse, so reset only the
            # backend session.  Explicit cold mode calls reset_warm_start().
            self._reset_session()
        row=dict(problem_index=len(self.rows),model=self.model.name,profile=self.profile,backend=self.backend,initialization='warm' if warm else 'cold',warm_start_requested=self.warm_strategy if self.backend=='cg2o' else self.ipopt_warm_mode,tracking_margin_applied_m=float(b.p.margin),base_clearance_at_x0_m=base_clearance,problem_sha256=problem_sha256,initial_sha256=initial_sha256,initial_max_eq_residual=initial['final_max_eq_residual'],initial_max_ineq_violation=initial['final_max_ineq_violation'],initial_obstacle_rows_violated=initial_obstacle_rows_violated,initial_max_obstacle_penetration_m=initial_max_obstacle_penetration,usable=usable,initialization_ms=initialization_ms,native_call_ms=native_call_ms,wall_ms=wall,**result)
        self.rows.append(row)
        Xfinal=b.states(z);slack_m=np.zeros(self.N);declared_slack_m=np.zeros(self.N)
        for k in range(self.N):
            distances=np.linalg.norm(padding[:,:3]-Xfinal[:3,k+1],axis=1)
            slack_m[k]=max(0.,np.max(padding[:,3]+b.p.margin-distances))
            declared_slack_m[k]=max(0.,np.max(padding[:,3]-distances))
        Sfinal=np.zeros((3,self.N));Sfinal[0]=declared_slack_m
        row['max_tracking_reserve_penetration_m']=float(np.max(slack_m))
        row['max_declared_safety_penetration_m']=float(np.max(declared_slack_m))
        if self.log:self.log.write(json.dumps(row)+'\n')
        return SolveResult(X=Xfinal,U=b.inputs(z),S=Sfinal,success=bool(result['converged']),usable=usable,
            status='converged' if result['converged'] else str(result['status']),iterations=result['iterations'],solve_ms=wall,
            gap_residual=result['final_max_eq_residual'],max_obstacle_slack=float(np.max(declared_slack_m)) if usable else float('inf'),max_arena_slack=0. if usable else float('inf'),max_dynamic_slack=0. if usable else float('inf'),cost=result['final_cost'],extra=row)
    def solve(self,x0,xref,uprev):
        r=self.solve_ocp(x0,xref[:3],xref[3:6],None,uprev=uprev)
        return r.U,r.X,r.success,r.solve_ms,r.status
