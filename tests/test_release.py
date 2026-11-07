import csv
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import cv2
from polish.dataset import PolishingVideoDataset, inside
from polish.paper_physics import PaperPhysicsLossBundle
from polish.data_integrity import audit
from polish.evaluate import paired_metrics
from polish.checkpoints import save_training_state, restore_training_state


def manifest(tmp_path, shared=False):
    (tmp_path/'dataset_info.json').write_text('{}')
    for index, split in enumerate(['train','val','test']):
        sid='same' if shared else f'sim{index}'
        frames=tmp_path/'frames'/sid
        frames.mkdir(parents=True,exist_ok=True)
        for i in range(3):
            image=np.full((16,32,3),index*35+i*3,dtype=np.uint8)
            cv2.imwrite(str(frames/f'frame_{i:06d}.png'),image)
        row={'sample_id':split,'source_dir':f'frames/{sid}','start_frame':0,'end_frame':3,'rpm':10,'raw_rpm':21,'slip':0.,'fps':100.,'regime':'controlled_c_slip','surface_type':'synthetic_c_slip','sim_id':sid,'simulation_seed':42 if shared else 42+index,'split':split}
        with (tmp_path/f'{split}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=row);w.writeheader();w.writerow(row)
    (tmp_path/'metadata.csv').write_text((tmp_path/'train.csv').read_text())
    return tmp_path/'metadata.csv'


def test_slip_zero_is_preserved_and_normalization_fixed(tmp_path):
    path=manifest(tmp_path)
    ds=PolishingVideoDataset(path,split='train',num_frames=3,resize_factor=1)
    assert ds[0]['physics_params'].tolist()==[0.,0.]
    assert ds[0]['video'].shape==(3,3,16,32)
    paper=PolishingVideoDataset(path,split='val',num_frames=3,resize_factor=1,rpm_column='raw_rpm',rpm_min=21,rpm_max=108)
    assert paper[0]['physics_params'][0]==0


def test_missing_split_does_not_fall_back(tmp_path):
    path=manifest(tmp_path);(tmp_path/'val.csv').unlink()
    with pytest.raises(FileNotFoundError):PolishingVideoDataset(path,split='val')


def test_real_records_rejected_before_frame_access(tmp_path):
    path=manifest(tmp_path)
    (tmp_path/'train.csv').write_text((tmp_path/'train.csv').read_text().replace('controlled_c_slip','original'))
    with pytest.raises(ValueError,match='synthetic'):PolishingVideoDataset(path,split='train')


def test_paths_cannot_escape(tmp_path):
    for value in ['../private','/private']:
        with pytest.raises(ValueError):inside(tmp_path,value)
    (tmp_path/'escape').symlink_to('/tmp')
    with pytest.raises(ValueError):inside(tmp_path,'escape/file')


def test_no_augmentation_on_test(tmp_path):
    with pytest.raises(ValueError,match='training'):PolishingVideoDataset(manifest(tmp_path),split='test',augment=True)


def test_missing_frames_fail(tmp_path):
    path=manifest(tmp_path)
    ds=PolishingVideoDataset(path,split='train',num_frames=3,resize_factor=1)
    (tmp_path/'frames/sim0/frame_000001.png').unlink()
    with pytest.raises(FileNotFoundError):ds[0]


def test_disjoint_audit(tmp_path):
    assert audit(manifest(tmp_path))['status']=='PASS'


def test_shared_trajectories_fail(tmp_path):
    report=audit(manifest(tmp_path,shared=True))
    assert report['status']=='FAIL'
    assert {'shared_simulation','shared_simulation_seed','frame_overlap'} <= {i['type'] for i in report['issues']}


def test_kinematic_limits_units_and_gradient():
    loss=PaperPhysicsLossBundle()
    rpm=torch.tensor([.0,.5,1.]);slip=torch.tensor([0.,.4,1.])
    veff,rt,wc,wb=loss.targets(rpm,slip)
    assert rt[0].item()==pytest.approx(1/(1+2**-.5))
    assert rt[-1].item()==1 and wb[-1].item()==0
    other=PaperPhysicsLossBundle(R_g=.6336,R_b=.02)
    assert torch.allclose(other.targets(rpm,slip)[0],veff*2)
    latent=torch.randn(4,3,2,4,4,requires_grad=True)
    total,metrics=loss(latent,torch.tensor([0.,0.,1.,1.]),torch.tensor([.1,.8,.1,.8]))
    total.backward()
    assert torch.isfinite(total) and torch.isfinite(latent.grad).all()


def test_strata_skip_and_detached_denominator():
    d=torch.tensor([1.,2.],requires_grad=True)
    rt=torch.tensor([.6,.9])
    loss=PaperPhysicsLossBundle.stratified_ratio(d,rt,torch.tensor([0.,0.]))
    loss.backward()
    expected=(2/2)*(d.detach()/1.5-rt/rt.mean())/1.5
    assert torch.allclose(d.grad,expected)
    skipped=PaperPhysicsLossBundle.stratified_ratio(d,rt,torch.tensor([0.,1.]))
    assert skipped.item()==0


def test_metrics_reject_shape_changes_and_missing_frames():
    a=np.zeros((3,16,32,3),dtype=np.uint8)
    b=np.full_like(a,10)
    metric=paired_metrics(a,b)
    assert metric['psnr_mean']==pytest.approx(28.1308036087)
    with pytest.raises(ValueError):paired_metrics(a,b[:2])


def test_optimizer_and_rng_roundtrip(tmp_path):
    import random
    model=torch.nn.Linear(2,1);opt=torch.optim.AdamW(model.parameters())
    model(torch.ones(1,2)).sum().backward();opt.step()
    config=SimpleNamespace(conditioning_mode='cross_attn',sc_encoding='fourier',num_frequencies_sc=16,physics_encoding='disentangled',method_profile='paper_declared',lora_rank=32,lora_alpha=32,output_width=32,output_height=16,num_frames=3)
    save_training_state(tmp_path,opt,2,3,1.,config)
    expected=(random.random(),np.random.random(),torch.rand(1))
    restored=restore_training_state(tmp_path,opt)
    actual=(random.random(),np.random.random(),torch.rand(1))
    assert restored['next_epoch']==3
    assert expected[:2]==actual[:2] and torch.equal(expected[2],actual[2])
