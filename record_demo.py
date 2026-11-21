#!/usr/bin/env python3
"""Record a fresh real run of Module 4 as a captioned application-output MP4.

Install requirements.txt and run install_sam2.py as documented in README.md.
python record_demo.py
python record_demo.py --show
python record_demo.py --checkpoint /path/to/sam2.1_hiera_tiny.pt

Both classical extraction and actual SAM2 inference execute during recording.
The recording saves their rendered outputs with timed reading holds. This is
not an OS desktop screen capture and has no audio or claimed student voice.
"""
from pathlib import Path
import argparse
import json
import textwrap
import time
import cv2
import numpy as np
import module4 as app

ROOT=Path(__file__).resolve().parent
FPS=20;SIZE=(1280,720)
BG=(248,251,253);INK=(58,46,31);BLUE=(80,62,24)


def line(frame,text,x,y,scale=.72,color=INK,bold=False):
    cv2.putText(frame,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,2 if bold else 1,cv2.LINE_AA)


def paragraph(frame,text,x,y,chars=96,scale=.72,gap=34):
    for row in textwrap.wrap(text,chars):line(frame,row,x,y,scale);y+=gap


def panel(title):
    frame=np.full((720,1280,3),BG,np.uint8)
    cv2.rectangle(frame,(0,0),(1279,106),BLUE,-1)
    line(frame,'MODULE 4 / '+title,36,48,.95,(255,255,255),True)
    line(frame,'Thermal dog boundary: classical OpenCV method + independent SAM2 comparison',36,83,.66,(235,229,218))
    cv2.rectangle(frame,(0,668),(1279,719),BLUE,-1)
    line(frame,'Actual computation outputs | Real credited FLIR image | Captioned application recording',28,700,.62,(255,255,255))
    return frame


def picture(frame,a,box):
    if a.ndim==2:a=cv2.cvtColor(np.uint8(a)*255 if a.dtype==bool else a,cv2.COLOR_GRAY2BGR)
    x,y,w,h=box
    factor=min(w/a.shape[1],h/a.shape[0])
    b=cv2.resize(a,(round(a.shape[1]*factor),round(a.shape[0]*factor)),interpolation=cv2.INTER_NEAREST)
    hh,ww=b.shape[:2];xx=x+(w-ww)//2;yy=y+(h-hh)//2
    frame[yy:yy+hh,xx:xx+ww]=b


def main(weights,show=False):
    before=json.loads((ROOT/'comparison_metrics.json').read_text()) if (ROOT/'comparison_metrics.json').exists() else None
    destination=ROOT/'Module_4_Working_Demo.mp4'
    writer=cv2.VideoWriter(str(destination),cv2.VideoWriter_fourcc(*'mp4v'),FPS,SIZE)
    if not writer.isOpened():raise RuntimeError('MP4 encoder unavailable')
    events=[];frames=0;start=time.perf_counter()

    def capture(frame,seconds,event):
        nonlocal frames
        n=round(FPS*seconds)
        events.append({'stage':event,'first_frame':frames,'frames':n,'execution_elapsed_seconds':time.perf_counter()-start})
        for _ in range(n):
            current=frame.copy();line(current,f'Playback {frames//FPS} s',1100,642,.47)
            writer.write(current);frames+=1
            if show:
                cv2.imshow('Module 4 - classical thermal boundary and SAM2',current)
                if cv2.waitKey(round(1000/FPS))&255==27:raise KeyboardInterrupt

    def progress(stage,data):
        if stage=='input':
            f=panel('Real thermal input and declared manual prompt')
            original=data['image'].copy();x0,y0,x1,y1=app.BOX
            cv2.rectangle(original,(x0,y0),(x1,y1),(0,220,255),2)
            cv2.circle(original,tuple(app.SEED),3,(255,100,0),-1)
            picture(f,original,(25,133,700,490))
            line(f,'640 x 512 FLIR A65 image',765,185,.75,BLUE,True)
            for j,row in enumerate(['Manual ROI:', '[270, 225, 340, 335]', 'Classical seed: (301, 284)', 'White-hot polarity', 'No temperature calibration']):
                line(f,row,765,252+j*57,.71)
            paragraph(f,'Public dataset photograph; not an Aya camera capture.',765,565,41,.66,30)
            capture(f,8,stage)
        elif stage=='classical':
            f=panel('Primary method: no machine learning')
            arrays=[data['roi'],data['blur'],data['thresholded'],data['mask'][225:335,270:340]]
            for i,(a,label) in enumerate(zip(arrays,['ROI','Gaussian blur','Otsu > 176','Cleaned component'])):
                line(f,label,34+i*315,155,.71,BLUE,True);picture(f,a,(23+i*315,179,285,350))
            paragraph(f,'5 x 5 smoothing -> Otsu threshold -> 3 x 3 opening and closing -> component containing the seed -> full-resolution external contour.',40,567,103,.72,33)
            line(f,f"Actual output: {data['metrics']['foreground_pixels']} foreground pixels; {data['metrics']['boundary_points']} ordered boundary points.",40,638,.72,BLUE,True)
            capture(f,11,stage)
        elif stage=='sam_ready':
            f=panel('Independent SAM2 model loaded')
            line(f,'Official SAM 2.1 Hiera Tiny checkpoint',55,200,1.0,BLUE,True)
            lines=['CPU float32 inference, four threads',
                   'Same original thermal image and same box prompt',
                   'No classical mask, contour, seed or threshold is supplied',
                   'Checkpoint SHA-256 checked before inference']
            for j,row in enumerate(lines):line(f,row,55,295+j*66,.82)
            paragraph(f,'The primary mask has already been saved. SAM2 is only the comparison method and cannot change that mask.',55,591,96,.73,32)
            capture(f,6,stage)
        elif stage=='sam2':
            f=panel('Actual SAM2 segmentation completed')
            picture(f,data['overlay'][220:340,260:350],(65,140,400,475))
            line(f,'Magenta: SAM2 boundary',565,197,.9,BLUE,True)
            line(f,f"Foreground: {data['metrics']['foreground_pixels']} pixels",565,276,.85)
            line(f,f"Predicted IoU score: {data['metrics']['predicted_iou_score_not_measured_accuracy']:.4f}",565,343,.84)
            paragraph(f,'The score is a model prediction. It is not measured accuracy against an annotated animal boundary.',565,411,49,.75,38)
            paragraph(f,'The raw mask, low-resolution logits, model settings and source hashes are saved.',565,565,51,.70,33)
            capture(f,9,stage)
        elif stage=='comparison':
            f=panel('Measured mask and boundary agreement')
            for i,(a,label) in enumerate(zip([data['overlay'],data['agreement']],['Both estimated outlines','White = shared foreground'])):
                line(f,label,38+i*330,151,.63,BLUE,True)
                picture(f,a[220:340,260:350],(35+i*330,175,300,410))
            m=data['metrics'];x=730
            for j,row in enumerate([f"IoU: {100*m['iou']:.2f}%",f"Dice: {100*m['dice']:.2f}%",f"Mean boundary distance: {m['mean_symmetric_boundary_distance_pixels']:.3f} px",f"95% Hausdorff distance: {m['symmetric_hd95_pixels']:.3f} px",f"SAM2-only pixels: {m['only_sam2_pixels']}"]):
                line(f,row,x,210+j*67,.73,BLUE,j<2)
            paragraph(f,'Green = classical. Magenta = SAM2. These metrics measure agreement, not ground-truth accuracy.',40,607,112,.66,30)
            capture(f,11,stage)

    try:
        f=panel('Working application demonstration')
        line(f,'Find the thermal animal boundary without a learned model.',50,190,.94,BLUE,True)
        for j,row in enumerate(['1. Execute classical thresholding and contour extraction.',
                                '2. Run SAM2 separately with the same image and box.',
                                '3. Measure overlap and boundary differences.',
                                '4. Inspect errors and document reproducibility.']):line(f,row,55,280+j*67,.80)
        paragraph(f,'This video records rendered outputs from a fresh execution with reading holds. It has no audio and is not an OS desktop capture.',55,582,99,.71,32)
        capture(f,6,'start')
        result=app.run_all(weights,progress)
        if before:
            for k in ['intersection_pixels','union_pixels','iou','dice']:
                np.testing.assert_allclose(result[k],before[k],rtol=0,atol=1e-12)
        f=panel('What the result establishes')
        for j,row in enumerate(['The classical mask covers the main bright body.',
                                'Dimmer outline details near ears, neck and legs differ.',
                                'No reference mask establishes pixel-exact accuracy.',
                                'The classical output was not retuned using SAM2.']):line(f,row,55,202+j*80,.86,BLUE,j==0)
        paragraph(f,'Image/prompt hashes, mask dimensions, classical independence, overlap formulas and identical-mask boundary metrics are checked in the code.',55,579,96,.72,33)
        capture(f,7,'validation')
        f=panel('Reproduce from the repository')
        for j,row in enumerate(['python module4.py all','python record_demo.py --show','python make_report.py']):line(f,row,60,216+j*83,1.,BLUE,True)
        paragraph(f,'Install requirements.txt and run install_sam2.py first. The official checkpoint downloads automatically on the first full run and is checksum-verified.',60,496,94,.77,36)
        line(f,'Source, masks and numerical evidence are included.',60,628,.64)
        capture(f,7,'finish')
    finally:
        writer.release()
        if show:cv2.destroyAllWindows()
    capture_video=cv2.VideoCapture(str(destination));decoded=0
    while True:
        ok,frame=capture_video.read()
        if not ok:break
        assert frame.shape[:2]==(720,1280);decoded+=1
    fps=capture_video.get(cv2.CAP_PROP_FPS);capture_video.release()
    assert decoded==frames and abs(fps-FPS)<1e-6
    app.save_json('DEMO_RECORDING_VALIDATION.json',{'recording_type':'Actual application-rendered outputs from a fresh classical and SAM2 execution; reading holds added.',
        'desktop_capture':False,'audio':False,'fps':FPS,'frame_count':frames,'decoded_frames':decoded,
        'size_pixels':list(SIZE),'duration_seconds':frames/FPS,'stages':events,'results_reproduced':True,
        'source_sha256':{name:app.sha(ROOT/name) for name in ['module4.py','record_demo.py']}})
    print(json.dumps({'duration_seconds':frames/FPS,'decoded_frames':decoded,'video':str(destination)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--show',action='store_true')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'sam2.1_hiera_tiny.pt')
    args=parser.parse_args();main(args.checkpoint,args.show)
