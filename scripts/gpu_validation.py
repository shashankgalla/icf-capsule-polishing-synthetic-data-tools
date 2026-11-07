"""Execute bounded synthetic integration checks with explicit local inputs."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--base-model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    dataset=args.dataset.resolve();out=args.output.resolve()
    if out.exists():raise FileExistsError('Diagnostic output must be new')
    out.mkdir(parents=True)
    env={**os.environ,'PYTHONPATH':str(root),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','WANDB_MODE':'disabled','CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    commands=[]
    def run(arguments,name):
        command=[sys.executable,*arguments]
        with (out/(name+'.log')).open('w') as log:
            done=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        commands.append({'command':command,'returncode':done.returncode,'log':name+'.log'})
        (out/'commands.json').write_text(json.dumps(commands,indent=2))
        if done.returncode:raise RuntimeError(f'{name} failed; inspect its log')
    metadata=dataset/'splits/trajectory_disjoint/metadata.csv'
    common=['-m','polish.train','--method_profile','paper_declared','--metadata_csv',str(metadata),'--pretrained_model',str(args.base_model),
        '--epochs','2','--batch_size','2','--grad_accum','1','--data_fraction','.006','--lora_rank','2','--lora_alpha','2','--num_frames','4',
        '--output_width','64','--output_height','32','--resize_factor','1','--clips_per_sample','1','--num_workers','0',
        '--sample_every_epoch','999','--eval_every_epoch','999','--save_every_epoch','1','--physics_warmup_epoch','-1','--physics_ramp_epochs','1',
        '--deterministic','--lambda_ratio','.1','--seed','42','--warmup_steps','0']
    run(common+['--output_dir',str(out/'uninterrupted')],'train_two_epochs')
    run(common+['--output_dir',str(out/'resumed'),'--resume_from',str(out/'uninterrupted/epoch_001')],'resume_second_epoch')
    a,b=out/'uninterrupted/final',out/'resumed/final'
    comparisons={}
    for relative in ['unet_lora/adapter_model.safetensors','physics_mlp.pt']:
        comparisons[relative]=hashlib.sha256((a/relative).read_bytes()).hexdigest()==hashlib.sha256((b/relative).read_bytes()).hexdigest()
    # Serialization can differ; compare tensors exactly as the authoritative test.
    import torch
    from safetensors.torch import load_file
    ad1,ad2=load_file(a/'unet_lora/adapter_model.safetensors'),load_file(b/'unet_lora/adapter_model.safetensors')
    comparisons['adapter_tensors_equal']=all(torch.equal(ad1[k],ad2[k]) for k in ad1)
    md1,md2=torch.load(a/'physics_mlp.pt',weights_only=True),torch.load(b/'physics_mlp.pt',weights_only=True)
    comparisons['encoder_tensors_equal']=all(torch.equal(md1[k],md2[k]) for k in md1)
    (out/'resume_comparison.json').write_text(json.dumps(comparisons,indent=2))
    row=next(csv.DictReader((metadata.parent/'test.csv').open()))
    image=dataset/row['source_dir']/f"frame_{int(row['start_frame']):06d}.png"
    infer=['-m','polish.cli','infer','--checkpoint',str(a),'--pretrained-model',str(args.base_model),'--image',str(image),'--rpm',row['raw_rpm'],
        '--slip',row['slip'],'--frames','4','--steps','2','--width','64','--height','32','--seed','42']
    run(infer+['--output',str(out/'inference_a.mp4')],'infer_a')
    run(infer+['--output',str(out/'inference_b.mp4')],'infer_b')
    import numpy as np
    comparisons['inference_arrays_equal']=np.array_equal(np.load(out/'inference_a.npy'),np.load(out/'inference_b.npy'))
    from polish.dataset import PolishingVideoDataset
    from polish.evaluate import paired_metrics
    ds=PolishingVideoDataset(metadata,split='test',num_frames=4,output_width=64,output_height=32,rpm_column='raw_rpm',rpm_min=21,rpm_max=108)
    reference=(ds[0]['video'].permute(0,2,3,1).numpy()*255).round().astype(np.uint8)
    metrics=paired_metrics(reference,np.load(out/'inference_a.npy'))
    (out/'integration_summary.json').write_text(json.dumps({'comparisons':comparisons,'paired_smoke_metrics':metrics,'frames':4,'resolution':[64,32],
        'epochs':2,'optimizer_steps':2,'adapter_rank':2,'paper_results_reproduced':False,'note':'Bounded synthetic execution test. No reported paper metric is claimed reproduced.'},indent=2))
    print(json.dumps({'comparisons':comparisons,'paired_smoke_metrics':metrics}))
    if not all(comparisons[k] for k in ['adapter_tensors_equal','encoder_tensors_equal','inference_arrays_equal']):
        raise SystemExit('Deterministic comparison failed')


if __name__=='__main__':main()
