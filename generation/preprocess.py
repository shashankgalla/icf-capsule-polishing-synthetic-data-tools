"""Apply a declared fixed crop to a new synthetic frame directory."""
import argparse
import json
from pathlib import Path
import cv2

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True,help='Synthetic PNG frame directory only')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--crop',type=int,nargs=4,required=True,metavar=('X','Y','WIDTH','HEIGHT'))
    p.add_argument('--size',type=int,nargs=2,default=[480,48])
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Output must be new')
    files=sorted(a.input.glob('*.png'))
    if not files:raise ValueError('No input frames')
    x,y,w,h=a.crop
    if min(x,y)<0 or min(w,h,*a.size)<=0:raise ValueError('Invalid crop or size')
    a.output.mkdir(parents=True)
    for i,path in enumerate(files):
        im=cv2.imread(str(path))
        if im is None or y+h>im.shape[0] or x+w>im.shape[1]:raise ValueError('Crop exceeds a readable frame')
        frame=cv2.resize(im[y:y+h,x:x+w],tuple(a.size),interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(a.output/f'frame_{i:06d}.png'),frame):raise OSError('Frame write failed')
    (a.output/'preprocessing.json').write_text(json.dumps({'crop':a.crop,'size':a.size,'interpolation':'INTER_AREA','frames':len(files),'fitted_statistics':False},indent=2))

if __name__=='__main__':main()
