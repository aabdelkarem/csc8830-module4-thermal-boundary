#!/usr/bin/env python3
"""CSc 8830 Module 4: classical animal boundaries and separate SAM2 comparison.

Install classical/report dependencies: python -m pip install -r requirements.txt
Classical ONLY (no ML imports): python module4.py classical
Install SAM2: follow README.md, then python module4.py all
Use an existing official checkpoint: python module4.py all --checkpoint PATH
Recompute comparison from saved masks: python module4.py compare

The primary method uses only OpenCV thresholding, morphology, connected
components and contours. SAM2 is an independent comparison, never an input
to the primary method. The declared ROI/seed were chosen before the SAM2 run.
The default photograph is a credited real public thermal image, not Aya's
camera capture. No ground-truth segmentation accuracy or Classroom turn-in
is claimed. All outputs are written next to this script.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
import urllib.request
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
IMAGE = ROOT/'thermal_dog.jpeg'
BOX = [270,225,340,335]  # x0,y0,x1,y1; classical crop is half-open.
SEED = [301,284]         # x,y; manual classical foreground seed.
MODEL = 'configs/sam2.1/sam2.1_hiera_t.yaml'
SAM_COMMIT = '2b90b9f5ceec907a1c18123530e92e794ad901a4'
CHECKPOINT_URL = 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt'
CHECKPOINT_SHA = '7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69'
REPO = 'https://github.com/aabdelkarem/csc8830-module4-thermal-boundary'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def save_json(name,data):
    (ROOT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def image():
    a=cv2.imread(str(IMAGE),cv2.IMREAD_COLOR)
    if a is None: raise FileNotFoundError(IMAGE)
    return a


def contours(mask):
    return cv2.findContours(np.uint8(mask)*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]


def select_component(binary):
    x0,y0,x1,y1=BOX
    element=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    opened=cv2.morphologyEx(binary,cv2.MORPH_OPEN,element)
    cleaned=cv2.morphologyEx(opened,cv2.MORPH_CLOSE,element)
    count,labels,stats,centers=cv2.connectedComponentsWithStats(cleaned,8)
    selected=int(labels[SEED[1]-y0,SEED[0]-x0])
    if selected==0: raise ValueError('The declared foreground seed became background.')
    return opened,cleaned,labels==selected,count-1


def classical(progress=None):
    emit=progress or (lambda *args:None)
    a=image(); gray=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
    x0,y0,x1,y1=BOX
    assert 0<=x0<x1<=a.shape[1] and 0<=y0<y1<=a.shape[0]
    emit('input',{'image':a,'box':BOX,'seed':SEED})
    start=time.perf_counter()
    blur=cv2.GaussianBlur(gray[y0:y1,x0:x1],(5,5),0)
    threshold,binary=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    opened,cleaned,selected,components=select_component(binary)
    mask=np.zeros_like(gray,dtype=bool);mask[y0:y1,x0:x1]=selected
    outline=contours(mask)
    assert len(outline)==1 and mask[SEED[1],SEED[0]]
    # Boundary extraction is exact for the ESTIMATED binary mask, not proof
    # that the estimated mask equals the unknown true animal silhouette.
    points=outline[0].reshape(-1,2)
    elapsed=time.perf_counter()-start
    overlay=a.copy();cv2.drawContours(overlay,outline,-1,(0,220,0),1)
    cv2.imwrite(str(ROOT/'classical_mask.png'),np.uint8(mask)*255)
    cv2.imwrite(str(ROOT/'classical_boundary.png'),overlay)
    np.savetxt(ROOT/'classical_boundary_xy.csv',points,delimiter=',',fmt='%d',header='x,y',comments='')
    sensitivity=[]
    for offset in [-20,-10,0,10,20]:
        t=float(np.clip(threshold+offset,0,255))
        candidate=cv2.threshold(blur,t,255,cv2.THRESH_BINARY)[1]
        try:
            variation=select_component(candidate)[2]
            union=np.count_nonzero(variation|selected)
            sensitivity.append({'threshold':t,'foreground_pixels':int(variation.sum()),
                                'iou_with_default_classical':float(np.count_nonzero(variation&selected)/union)})
        except ValueError:
            sensitivity.append({'threshold':t,'foreground_pixels':0,'iou_with_default_classical':0.})
    metrics={'image_sha256':sha(IMAGE),'image_width_height':[a.shape[1],a.shape[0]],
             'manual_roi_xyxy':BOX,'manual_seed_xy':SEED,'palette':'white_hot_inferred_visually',
             'otsu_threshold':float(threshold),'foreground_pixels':int(mask.sum()),
             'external_contours':len(outline),'boundary_points':len(points),
             'contour_perimeter_pixels':float(cv2.arcLength(outline[0],True)),
             'gaussian_kernel':[5,5],'morphology_kernel':'3x3 ellipse; open then close',
             'components_after_morphology':components,'algorithm_seconds_single_run':elapsed,
             'mask_sha256':sha(ROOT/'classical_mask.png'),'machine_learning_used':False,
             'threshold_sensitivity':sensitivity}
    save_json('classical_metrics.json',metrics)
    emit('classical',{'image':a,'roi':gray[y0:y1,x0:x1],'blur':blur,'thresholded':binary,
                      'cleaned':cleaned,'mask':mask,'overlay':overlay,'metrics':metrics})
    return mask,metrics


def checkpoint(path):
    path=Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.download')
        print('Downloading official SAM2.1 tiny checkpoint (156 MB)...',flush=True)
        urllib.request.urlretrieve(CHECKPOINT_URL,temporary)
        if sha(temporary)!=CHECKPOINT_SHA: raise ValueError('Checkpoint checksum mismatch.')
        temporary.replace(path)
    if sha(path)!=CHECKPOINT_SHA: raise ValueError('Expected the official SAM2.1 hiera tiny checkpoint.')
    return path


def sam2(checkpoint_path,progress=None):
    # These imports exist only in the independent comparison branch.
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    emit=progress or (lambda *args:None)
    a=image(); weights=checkpoint(checkpoint_path)
    torch.manual_seed(8830);torch.set_num_threads(4)
    start=time.perf_counter()
    model=build_sam2(MODEL,str(weights),device='cpu',mode='eval',apply_postprocessing=True)
    predictor=SAM2ImagePredictor(model,mask_threshold=0.,max_hole_area=0.,max_sprinkle_area=0.)
    loaded=time.perf_counter()
    emit('sam_ready',{'image':a,'model':'SAM 2.1 Hiera Tiny','box':BOX})
    inference_start=time.perf_counter()
    with torch.inference_mode():
        predictor.set_image(cv2.cvtColor(a,cv2.COLOR_BGR2RGB))
        masks,scores,logits=predictor.predict(box=np.asarray(BOX,dtype=np.float32),multimask_output=False)
    elapsed=time.perf_counter()-inference_start
    mask=np.asarray(masks)[0].astype(bool)
    assert mask.shape==a.shape[:2] and mask.any()
    cv2.imwrite(str(ROOT/'sam2_mask.png'),np.uint8(mask)*255)
    np.savez_compressed(ROOT/'sam2_raw_prediction.npz',masks=masks,scores=scores,low_resolution_logits=logits)
    overlay=a.copy();cv2.drawContours(overlay,contours(mask),-1,(230,0,230),1)
    cv2.imwrite(str(ROOT/'sam2_boundary.png'),overlay)
    metrics={'image_sha256':sha(IMAGE),'model':'SAM 2.1 Hiera Tiny','upstream_commit':SAM_COMMIT,
             'model_config':MODEL,'checkpoint_url':CHECKPOINT_URL,'checkpoint_sha256':sha(weights),
             'prompt_box_xyxy':BOX,'prompt_points':None,'mask_input':None,'multimask_output':False,
             'device':'cpu','dtype':'float32','cpu_threads':4,'torch_version':torch.__version__,
             'predicted_iou_score_not_measured_accuracy':float(np.asarray(scores).ravel()[0]),
             'foreground_pixels':int(mask.sum()),'mask_sha256':sha(ROOT/'sam2_mask.png'),
             'model_load_seconds_single_run':loaded-start,'inference_seconds_single_run':elapsed,
             'dynamic_multimask_via_stability':True,'hole_or_sprinkle_cleanup':False}
    save_json('sam2_metrics.json',metrics)
    emit('sam2',{'image':a,'mask':mask,'overlay':overlay,'metrics':metrics})
    return mask,metrics


def overlap(a,b):
    a,b=np.asarray(a,dtype=bool),np.asarray(b,dtype=bool)
    intersection=int(np.count_nonzero(a&b));union=int(np.count_nonzero(a|b))
    total=int(a.sum()+b.sum())
    return {'intersection_pixels':intersection,'union_pixels':union,
            'iou':intersection/union if union else 1.,'dice':2*intersection/total if total else 1.,
            'only_classical_pixels':int(np.count_nonzero(a&~b)),
            'only_sam2_pixels':int(np.count_nonzero(b&~a))}


def boundary_distances(a,b):
    edges=[]
    for mask in [a,b]:
        edge=np.zeros(mask.shape,np.uint8)
        cv2.drawContours(edge,contours(mask),-1,1,1);edges.append(edge.astype(bool))
    ea,eb=edges
    da=cv2.distanceTransform(np.uint8(~ea),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    db=cv2.distanceTransform(np.uint8(~eb),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    a_to_b,b_to_a=db[ea],da[eb]
    precision=float((a_to_b<=2.).mean());recall=float((b_to_a<=2.).mean())
    return {'mean_symmetric_boundary_distance_pixels':float((a_to_b.mean()+b_to_a.mean())/2),
            'symmetric_hd95_pixels':float(max(np.percentile(a_to_b,95),np.percentile(b_to_a,95))),
            'hausdorff_max_pixels':float(max(a_to_b.max(),b_to_a.max())),
            'boundary_f1_within_2_pixels':2*precision*recall/(precision+recall) if precision+recall else 0.}


def compare(progress=None):
    emit=progress or (lambda *args:None)
    cm=json.loads((ROOT/'classical_metrics.json').read_text())
    sm=json.loads((ROOT/'sam2_metrics.json').read_text())
    assert cm['image_sha256']==sm['image_sha256']==sha(IMAGE)
    assert cm['manual_roi_xyxy']==sm['prompt_box_xyxy']==BOX
    assert cm['mask_sha256']==sha(ROOT/'classical_mask.png')
    assert sm['mask_sha256']==sha(ROOT/'sam2_mask.png')
    a=cv2.imread(str(ROOT/'classical_mask.png'),0)>0
    b=cv2.imread(str(ROOT/'sam2_mask.png'),0)>0
    assert a.shape==b.shape==image().shape[:2]
    # Hand-verifiable toy sets independently validate the overlap formula.
    toy=overlap([1,1,0,0],[0,1,1,0])
    assert abs(toy['iou']-1/3)<1e-12 and abs(toy['dice']-.5)<1e-12
    identical=boundary_distances(a,a)
    assert identical['hausdorff_max_pixels']==0 and identical['boundary_f1_within_2_pixels']==1
    result={**overlap(a,b),**boundary_distances(a,b),
            'comparison_type':'Agreement of two estimates, not ground-truth accuracy.',
            'same_original_image_and_box':True,'full_resolution_comparison':True,
            'classical_executed_independently':cm['machine_learning_used'] is False,
            'validation_checks_passed':True}
    save_json('comparison_metrics.json',result)
    overlay=image();cv2.drawContours(overlay,contours(a),-1,(0,220,0),1)
    cv2.drawContours(overlay,contours(b),-1,(230,0,230),1)
    cv2.imwrite(str(ROOT/'comparison_boundaries.png'),overlay)
    diff=np.zeros((*a.shape,3),np.uint8)
    diff[a&b]=(230,230,230);diff[a&~b]=(0,220,0);diff[b&~a]=(230,0,230)
    cv2.imwrite(str(ROOT/'mask_agreement.png'),diff)
    emit('comparison',{'image':image(),'classical':a,'sam2':b,'overlay':overlay,'agreement':diff,'metrics':result})
    return result


def run_all(checkpoint_path,progress=None):
    primary,cm=classical(progress)
    before=sha(ROOT/'classical_mask.png')
    sam2(checkpoint_path,progress)
    assert before==sha(ROOT/'classical_mask.png') # Comparator never modifies primary result.
    result=compare(progress)
    versions={}
    for package in ['numpy','opencv-python','torch','torchvision','SAM-2','hydra-core','omegaconf','Pillow','reportlab']:
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]='not installed'
    save_json('TESTED_ENVIRONMENT.json',{'python':platform.python_version(),'platform':platform.platform(),'packages':versions})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['classical','sam2','compare','all'])
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'sam2.1_hiera_tiny.pt')
    args=parser.parse_args()
    if args.command=='classical': result=classical()[1]
    elif args.command=='sam2': result=sam2(args.checkpoint)[1]
    elif args.command=='compare': result=compare()
    else: result=run_all(args.checkpoint)
    print(json.dumps(result,indent=2))
