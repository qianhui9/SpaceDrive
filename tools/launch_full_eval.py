"""Detach or reconnect to the full nuScenes evaluation worker."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--assets',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=Path(args.output_dir).resolve()
    output.mkdir(parents=True,exist_ok=True)
    pid_file=output/'worker.json'
    if pid_file.exists():
        existing=json.loads(pid_file.read_text())
        process_path=Path('/proc')/str(existing['pid'])
        try:
            command=(process_path/'cmdline').read_bytes().replace(b'\x00',b' ').decode()
        except OSError:
            command=''
        if 'infer_nuscenes_full.py' in command and str(output) in command:
            print('ALREADY_RUNNING '+json.dumps(existing),flush=True)
            return
    status_file=output/'status.json'
    if status_file.exists() and json.loads(status_file.read_text()).get('stage')=='completed':
        print('ALREADY_COMPLETED '+str(output/'metrics.json'),flush=True)
        return
    command=[sys.executable,'-u',str(root/'tools/infer_nuscenes_full.py'),
             '--checkpoint',str(Path(args.checkpoint).resolve()),'--data-root',str(Path(args.data_root).resolve()),
             '--assets',str(Path(args.assets).resolve()),'--output-dir',str(output),'--resume']
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
    env['MPLCONFIGDIR']=str(output/'matplotlib')
    with (output/'inference.log').open('a') as stream:
        stream.write('\nLAUNCH '+time.strftime('%Y-%m-%d %H:%M:%S')+' '+json.dumps(command)+'\n')
        stream.flush()
        process=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    record=dict(pid=process.pid,command=command,log=str(output/'inference.log'),status=str(output/'status.json'),launched_at=time.time())
    temporary=pid_file.with_suffix('.tmp')
    temporary.write_text(json.dumps(record,indent=2))
    temporary.replace(pid_file)
    print('LAUNCHED '+json.dumps(record),flush=True)


if __name__=='__main__':
    main()
