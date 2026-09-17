"""Discover robot chain, matching SRDF group and active trajectory controller."""
import numpy as np
import xml.etree.ElementTree as ET


def chain_joints(root, base, tip):
    joints={j.find('child').get('link'):j for j in root.findall('joint')}
    chain=[];seen=set()
    while tip!=base:
        if tip in seen or tip not in joints:
            raise ValueError(f'No robot chain from {base} to {tip}; check the calibration robot frames.')
        seen.add(tip);joint=joints[tip]
        if joint.get('type')!='fixed':
            if joint.get('type')!='revolute' or joint.find('mimic') is not None:
                raise ValueError('Automatic calibration currently requires bounded independent revolute joints.')
            chain.append(joint.get('name'))
        tip=joint.find('parent').get('link')
    return list(reversed(chain))


def matching_group(urdf, semantic, names, requested=''):
    groups={g.get('name'):g for g in ET.fromstring(semantic).findall('group')}
    def expand(name,seen):
        if name in seen or name not in groups: raise ValueError('Invalid SRDF group structure.')
        group=groups[name];result={j.get('name') for j in group.findall('joint')}
        for chain in group.findall('chain'):
            result.update(chain_joints(urdf,chain.get('base_link'),chain.get('tip_link')))
        for child in group.findall('group'):result.update(expand(child.get('name'),seen|{name}))
        # Fixed joints are not controlled DOFs.
        return {n for n in result if urdf.find(f"joint[@name='{n}']").get('type')!='fixed'}
    matches=[]
    for name in groups:
        try:
            if expand(name,set())==set(names): matches.append(name)
        except (ValueError, AttributeError):
            continue
    if requested:
        if requested not in matches: raise ValueError('Configured MoveIt group does not match the calibration robot chain.')
        return requested
    if len(matches)!=1: raise ValueError('Cannot identify one MoveIt group for this chain; configure auto_group explicitly.')
    return matches[0]


def matching_controller(controllers,names):
    matches=[]
    for c in controllers:
        claimed={item.rsplit('/',1)[0] for item in c.claimed_interfaces}
        if c.state=='active' and 'JointTrajectoryController' in c.type and claimed==set(names):
            matches.append('/'+c.name.strip('/')+'/follow_joint_trajectory')
    if len(matches)!=1:raise ValueError('Cannot identify one active trajectory controller; configure auto_controller explicitly.')
    return matches[0]
