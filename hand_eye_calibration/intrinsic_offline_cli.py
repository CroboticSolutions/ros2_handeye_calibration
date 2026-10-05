"""Offline Intrinsic-style pose solve; never writes or applies an active calibration."""
import argparse
import json
import numpy as np
from . import calibration_dataset, intrinsic_solver as solver


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset');parser.add_argument('--bootstrap',type=int,default=40)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    data=calibration_dataset.load(args.dataset)
    training=[s for s in data['samples'] if s.get('role','training')=='training']
    validation=[s for s in data['samples'] if s.get('role')=='validation']
    robot=[s['robot'] for s in training];tracking=[s['tracking'] for s in training]
    report=solver.solve(robot,tracking)
    report['uncertainty']=solver.uncertainty(robot,tracking,report,args.bootstrap)
    if validation:
        report['validation_views']=solver.metrics(np.array([solver.matrix(s['robot']) for s in validation]),
                                                  np.array([solver.matrix(s['tracking']) for s in validation]),
                                                  solver.matrix(report['transform']),solver.matrix(report['board_in_base']))
    report['acceptance']=solver.evaluate(report)
    with open(args.output,'w') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({'solver':report['solver'],'views':len(training),'metrics':report['pose_metrics'],
                      'uncertainty':report['uncertainty'],'acceptance':report['acceptance']},indent=2))
    return 0 if report['acceptance']['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
