"""Evaluate actual old-workflow flights in one physical frame and clock.

No controller changes. Gazebo truth is used only after the flight.
"""
import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from pyulog import ULog
from scipy.signal import savgol_filter
from px4_nmpc_core.math_utils import quat_wxyz_to_rotation,quat_wxyz_to_euler
from px4_nmpc_core.scenario import controller_origin_ned_of
ROOT=Path(__file__).resolve().parents[2]


def evaluate(folder):
    folder=Path(folder);scene=json.loads((folder/'runtime/scenario.json').read_text());origin=controller_origin_ned_of(scene)
    log=folder/'flight.ulg'
    if not log.exists():
        text=(folder/'logs/px4.log').read_text(errors='replace');name=re.search(r'(?:\./)?log/(\d{4}-\d+-\d+/[^\s]+\.ulg)',text)
        if not name:raise ValueError('No actual PX4 ULog path in flight log')
        shutil.copy2(ROOT/'build/externals/PX4-Autopilot/px4_sitl_default/rootfs/log'/name.group(1),log)
    flight=np.genfromtxt(folder/'flight.csv',delimiter=',',names=True)
    odom_names=['pN','pE','pD','vN','vE','vD'];raw=np.vstack([flight[n] for n in odom_names]).T
    lookup={tuple(np.round(row,5)):float(t) for row,t in zip(raw,flight['t']) if np.linalg.norm(row[3:])>.3}
    u=ULog(str(log),message_name_filter_list=['vehicle_local_position','vehicle_local_position_groundtruth','vehicle_attitude_groundtruth'])
    local=u.get_dataset('vehicle_local_position').data;gt=u.get_dataset('vehicle_local_position_groundtruth').data;att=u.get_dataset('vehicle_attitude_groundtruth').data
    matches=[]
    for i,stamp in enumerate(local['timestamp']):
        key=tuple(round(float(local[k][i]),5) for k in ['x','y','z','vx','vy','vz'])
        if key in lookup:matches.append((float(stamp)/1e6,lookup[key]))
    if len(matches)<10:raise ValueError('Insufficient matched uORB/DDS samples')
    boot,relative=np.array(matches).T;simt=gt['timestamp'].astype(float)/1e6;t=np.interp(simt,boot,relative)
    pose=np.fromstring(ET.parse(ROOT/'externals/PX4-Autopilot/Tools/simulation/gz/models/x500_base/model.sdf').getroot().find('model/pose').text,sep=' ')
    if not np.allclose(pose[3:],0):raise ValueError('Unexpected merged x500 model rotation')
    offset=pose[:3]*[1,-1,-1]
    pos=np.vstack([gt[k] for k in ['x','y','z']]).astype(float);pos[:2]+=origin[:2,None]
    angles=np.zeros_like(pos);iq=np.searchsorted(att['timestamp'],gt['timestamp']).clip(0,len(att['timestamp'])-1)
    for j,i in enumerate(iq):
        q=[att[f'q[{k}]'][i] for k in range(4)];pos[:,j]+=quat_wxyz_to_rotation(q)@offset;angles[:,j]=quat_wxyz_to_euler(q)
    diag=list(csv.DictReader((folder/'solves.csv').open()));nav=min(float(r['t']) for r in diag if r['phase']=='NMPC');end=min([float(r['t']) for r in diag if r['phase']=='GOAL'] or [float(t[-1])])
    selected=(t>=nav)&(t<=end);distinct=np.r_[True,np.diff(simt)>1e-7];vel=np.gradient(pos[:,distinct],simt[distinct],axis=1)
    moving=(t[distinct]>=nav)&(t[distinct]<=end)
    obstacles=np.array(scene['obstacle_centers_ned']);radii=np.array(scene['obstacle_radii']);radius=.35
    clearance=lambda X:np.min(np.linalg.norm(X.T[:,None,:]-obstacles[None,:,:],axis=2)-radii[None,:]-radius,axis=1)
    uniform=np.arange(simt[selected][0],simt[selected][-1],.01);resampled=np.array([np.interp(uniform,simt[selected],row[selected]) for row in pos])
    jerk=savgol_filter(resampled,25,3,deriv=3,delta=.01,axis=1)[:,12:-12]
    result=dict(environment='actual PX4 SITL + Gazebo, original feature branch runtime',outcome=json.loads((folder/'summary.json').read_text())['outcome'],
        time_to_goal_wall_s=end-nav,minimum_true_body_clearance_m=float(clearance(pos[:,selected]).min()),
        maximum_true_speed_mps=float(np.linalg.norm(vel[:,moving],axis=0).max()),maximum_circular_tilt_deg=float(np.rad2deg(np.linalg.norm(angles[:2,selected],axis=0).max())),
        yaw_span_deg=float(np.rad2deg(np.ptp(np.unwrap(angles[2,selected])))),path_length_m=float(np.linalg.norm(np.diff(pos[:,selected],axis=1),axis=0).sum()),
        lateral_span_m=float(np.ptp(pos[1,selected])),vertical_span_m=float(np.ptp(pos[2,selected])),filtered_jerk_rms_mps3=float(np.sqrt(np.mean(np.sum(jerk*jerk,axis=0)))),
        smoothness_method='100 Hz resampling, 250 ms cubic Savitzky-Golay third derivative; 120 ms edge samples excluded',
        safety_margin_met=bool(clearance(pos[:,selected]).min()>=.35),matched_samples=len(matches),model_to_body_frd=offset.tolist(),
        evaluation='post-flight truth only; no truth feedback or frame changes in controller')
    estimated=raw[:,:3].T+origin[:,None]
    np.savez_compressed(folder/'truth.npz',t=t-nav,simt=simt,position=pos,rpy=angles,navigation=selected,estimate=estimated,estimate_t=flight['t']-nav)
    (folder/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folders',type=Path,nargs='+');a=p.parse_args()
    for folder in a.folders:print(folder,json.dumps(evaluate(folder),indent=2))
