#!/usr/bin/env python3
"""Build the Module 4 figures and PDF from already executed results.

python -m pip install -r requirements.txt
python module4.py all                 # first execute both methods
python make_report.py

Reads the actual saved masks/metrics. It neither invents SAM2 predictions
nor reruns inference. The generated PDF links to the assignment repository.
"""
from pathlib import Path
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
import module4 as app

ROOT=Path(__file__).resolve().parent
ASSIGNMENT='https://docs.google.com/document/d/1HTuvFDvvbDXWJSi9XztM3WnG6kZDAs-iY63GTgmDjsM/edit'
NAVY=colors.HexColor('#183e50');ORANGE=colors.HexColor('#9a4a24')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=10,leading=14,textColor=colors.HexColor('#20313a'),spaceAfter=9))
styles.add(ParagraphStyle(name='SmallCustom',parent=styles['BodyCustom'],fontSize=8.5,leading=11,spaceAfter=7))
styles.add(ParagraphStyle(name='TitleCustom',fontName='Helvetica-Bold',fontSize=20,leading=24,textColor=NAVY,spaceAfter=9))
styles.add(ParagraphStyle(name='SubCustom',fontName='Helvetica-Bold',fontSize=12,leading=16,textColor=NAVY,spaceBefore=7,spaceAfter=7))


def p(text,small=False):return Paragraph(text,styles['SmallCustom' if small else 'BodyCustom'])
def heading(text):return Paragraph(text,styles['SubCustom'])


def table(rows,widths):
    cells=[[p(str(v),True) for v in row] for row in rows]
    t=Table(cells,colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9f0f3')),
                          ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),
                          ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),
                          ('BOTTOMPADDING',(0,0),(-1,-1),2),
                          ('LINEBELOW',(0,0),(-1,-1),.4,colors.HexColor('#cbd8de'))]))
    return t


def picture(name,width,height):return Image(str(ROOT/name),width=width,height=height)


def figures():
    a=app.image();x0,y0,x1,y1=app.BOX
    declared=a.copy();cv2.rectangle(declared,(x0,y0),(x1,y1),(0,220,255),2)
    cv2.circle(declared,tuple(app.SEED),3,(255,100,0),-1)
    cv2.imwrite(str(ROOT/'declared_input.png'),declared)
    roi=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)[y0:y1,x0:x1]
    blur=cv2.GaussianBlur(roi,(5,5),0)
    t,binary=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    clean=app.select_component(binary)[1]
    mask=cv2.imread(str(ROOT/'classical_mask.png'),0)[y0:y1,x0:x1]
    fig,axes=plt.subplots(1,4,figsize=(10,4))
    for ax,field,title in zip(axes,[roi,blur,binary,mask],['Declared ROI','5 x 5 Gaussian',f'Otsu > {t:g}','Cleaned component']):
        ax.imshow(field,cmap='gray',vmin=0,vmax=255,interpolation='nearest');ax.set_title(title,fontsize=12);ax.axis('off')
    fig.subplots_adjust(left=.02,right=.99,bottom=.03,top=.88,wspace=.08)
    fig.savefig(ROOT/'classical_stages.png',dpi=150,bbox_inches='tight',pad_inches=.12);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(10,4))
    for ax,name,title in zip(axes,['thermal_dog.jpeg','classical_boundary.png','sam2_boundary.png','mask_agreement.png'],
                              ['Thermal input','Classical: green','SAM2: magenta','White = overlap']):
        field=cv2.cvtColor(cv2.imread(str(ROOT/name)),cv2.COLOR_BGR2RGB)[220:340,260:350]
        ax.imshow(field,interpolation='nearest');ax.set_title(title,fontsize=12);ax.axis('off')
    fig.subplots_adjust(left=.02,right=.99,bottom=.03,top=.88,wspace=.08)
    fig.savefig(ROOT/'comparison_zoom.png',dpi=150,bbox_inches='tight',pad_inches=.12);plt.close(fig)


def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#cbd8de'));canvas.line(44,42,A4[0]-44,42)
    canvas.setFont('Helvetica',8);canvas.setFillColor(NAVY)
    canvas.drawString(44,29,'Aya Abdelkarem | CSc 8830 | Module 4')
    canvas.drawRightString(A4[0]-44,29,str(doc.page))


def main():
    cm=json.loads((ROOT/'classical_metrics.json').read_text())
    sm=json.loads((ROOT/'sam2_metrics.json').read_text())
    m=app.compare() # Verify hashes/inputs and recompute numerical evidence.
    figures()
    story=[]
    def add(*items):story.extend(items)
    add(Paragraph('Module 4: Thermal animal boundaries',styles['TitleCustom']),
        p('Classical OpenCV extraction with an independent SAM2 comparison'),
        p('<b>Objective.</b> Extract the boundary of the dog in a thermal image using a method with no machine learning or deep learning, then compare its result with SAM2.'),
        p(f'<b>Repository:</b> <link href="{app.REPO}" color="#166a9b">{app.REPO}</link>',True),
        picture('declared_input.png',360,288),
        p('Original 640 x 512 FLIR image. Yellow rectangle: declared animal ROI. Blue point: classical foreground seed. The nearby person is outside the selected ROI.',True),
        heading('Data and manual inputs'),
        p('The unchanged public FLIR A65 photograph is credited to sensationTI/FLIR_IR_Expansion, original path misc/FLIR_08863.jpeg. It is a real thermal dataset image, not a photograph captured by Aya. Image attribution and the source repository license accompany the code.'),
        p('The fixed ROI is [270,225,340,335] in x0,y0,x1,y1 coordinates; the foreground seed is (301,284). These were chosen before the SAM2 comparison. The primary method is manually initialized, not an automatic animal detector.'),
        p('White-hot polarity is inferred visually. Brightness is not treated as a calibrated temperature measurement. The dog occupies a small part of the original image, so one-pixel changes noticeably affect the outline.'))
    add(PageBreak(),Paragraph('Classical method and boundary output',styles['TitleCustom']),
        p('The primary branch imports no neural network model and uses no learned weights. Its output is saved before SAM2 is evaluated.'),
        picture('classical_stages.png',500,200),
        heading('Algorithm'),
        p('<b>1.</b> Convert the original image to grayscale and crop the declared ROI.<br/><b>2.</b> Apply a 5 x 5 Gaussian filter to reduce noise.<br/><b>3.</b> Use Otsu thresholding to split the ROI histogram into foreground/background.<br/><b>4.</b> Open and then close with a 3 x 3 elliptical structuring element.<br/><b>5.</b> Keep the connected component containing the declared foreground seed.<br/><b>6.</b> Map the mask back to the full image and trace its external contour using CHAIN_APPROX_NONE.'),
        p('<b>Otsu criterion.</b> Select T that maximizes w0(T) w1(T) [mu0(T)-mu1(T)]^2, where the two weights are class proportions and the two means are class intensities. The white-hot foreground is I(x,y) &gt; T. This is histogram optimization, not a learned classifier.'),
        table([['Executed quantity','Value'],['Otsu threshold',f"{cm['otsu_threshold']:g} / 255"],
               ['Foreground pixel count',cm['foreground_pixels']],['Ordered external boundary points',cm['boundary_points']],
               ['Contour perimeter',f"{cm['contour_perimeter_pixels']:.3f} pixels"]],[270,230]),
        Spacer(1,11),p('classical_boundary_xy.csv contains all ordered contour vertices as x,y image coordinates. classical_mask.png stores the estimated foreground, and classical_boundary.png draws its outline in green.',True),
        p('<b>Meaning of exact boundary.</b> The script extracts the complete discrete external boundary of its estimated binary mask. There is no pixel-level reference annotation establishing that this is the exact physical animal silhouette. Smoothing, thresholding and morphology can remove ears, thin limbs and dim outer pixels.'))
    add(PageBreak(),Paragraph('Independent SAM2 comparison',styles['TitleCustom']),
        p('SAM 2.1 Hiera Tiny is used only as the requested independent comparator. It receives the unchanged full RGB-converted thermal image and the same box prompt. It receives no classical mask, seed, contour or threshold. CPU float32 inference uses four threads and the official checkpoint.'),
        picture('comparison_zoom.png',500,200),
        p('Same zoom for all panels. Green: classical-only outline or pixels; magenta: SAM2-only outline or pixels; white in the final panel: shared foreground. Both masks are compared at the original 640 x 512 resolution.',True),
        table([['Measured agreement','Value'],['Classical / SAM2 foreground pixels',f"{cm['foreground_pixels']} / {sm['foreground_pixels']}"],
               ['Intersection / union pixels',f"{m['intersection_pixels']} / {m['union_pixels']}"],
               ['Intersection-over-union (IoU)',f"{m['iou']:.6f} ({m['iou']*100:.2f}%)"],
               ['Dice overlap',f"{m['dice']:.6f} ({m['dice']*100:.2f}%)"],
               ['Mean symmetric boundary distance',f"{m['mean_symmetric_boundary_distance_pixels']:.3f} pixels"],
               ['Symmetric 95th-percentile Hausdorff distance',f"{m['symmetric_hd95_pixels']:.3f} pixels"],
               ['Boundary F1 within 2 pixels',f"{m['boundary_f1_within_2_pixels']:.6f}"]],[330,170]),
        Spacer(1,10),p(f'<b>Worked calculation.</b> IoU = |A intersect B| / |A union B| = {m["intersection_pixels"]}/{m["union_pixels"]} = {m["iou"]:.6f}. Dice = 2|A intersect B|/(|A|+|B|) = {2*m["intersection_pixels"]}/({cm["foreground_pixels"]}+{sm["foreground_pixels"]}) = {m["dice"]:.6f}.',True),
        p(f'All {cm["foreground_pixels"]} classical foreground pixels are also in SAM2; SAM2 includes {m["only_sam2_pixels"]} additional pixels. Visual differences concentrate near dimmer outline details, including ears, the neck and legs. The classical method was not retuned to match SAM2.',True),
        p(f'<b>Interpretation.</b> These numbers measure agreement between two estimates, not accuracy. SAM2\'s predicted IoU score ({sm["predicted_iou_score_not_measured_accuracy"]:.4f}) is a model prediction, not an observed score against ground truth.',True))
    add(PageBreak(),Paragraph('Validation and reproduction',styles['TitleCustom']),
        heading('Sensitivity of the classical estimate'),
        p('A fixed, disclosed threshold sweep shows that the boundary depends on intensity selection. The submitted primary output remains the automatically chosen Otsu threshold; SAM2 is not used to select among these alternatives.',True),
        table([['Threshold','Foreground pixels','IoU with default classical mask']]+[[f"{r['threshold']:g}",r['foreground_pixels'],f"{r['iou_with_default_classical']:.4f}"] for r in cm['threshold_sensitivity']],[110,160,230]),
        heading('Checks performed'),
        p('Image SHA-256 and box equality are checked before comparison. Both binary masks have original-image dimensions and valid checksums. The classical foreground contains its seed and has one external contour. The comparator leaves the classical mask unchanged. A toy overlap example independently checks the IoU/Dice formulas; identical-mask boundary checks return zero distance and boundary F1 = 1.',True),
        heading('Run from a fresh repository download'),
        p('Use Python 3.10 or later in a virtual environment. Full SAM2 inference needs an initial network download of software and the official 156 MB checkpoint; the script verifies the checkpoint SHA-256. The saved results are already included.',True),
        p('<font face="Courier">python -m pip install -r requirements.txt<br/>python install_sam2.py<br/>python module4.py all<br/>python record_demo.py<br/>python make_report.py</font>',True),
        p('For the non-ML primary result alone, run <font face="Courier">python module4.py classical</font>. For the visible application, use <font face="Courier">python record_demo.py --show</font>. The MP4 records actual rendered outputs from a fresh execution, with captions and reading holds; it is not an OS desktop capture and has no audio.',True),
        heading('Limits and provenance'),
        p('One image and one manually selected animal are evaluated. No general animal-detection accuracy, calibrated temperature, metric body area or pixel-exact ground truth is available. Thermal contrast, occlusion, touching warm objects and thin anatomy remain failure cases. SAM2 is also an estimate and can fail under domain shift.',True),
        p('Review the implementation and results before submission. No code submits work to Google Classroom. Library versions, input/model hashes, prediction arrays, masks, contours and recording validation are included.',True),
        heading('References'),
        p(f'<link href="{ASSIGNMENT}" color="#166a9b">Module 4 assignment</link> | <link href="https://github.com/sensationTI/FLIR_IR_Expansion" color="#166a9b">Thermal-image source</link><br/><link href="https://github.com/facebookresearch/sam2" color="#166a9b">Meta SAM2 source, checkpoint and image-predictor API</link><br/><link href="https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html" color="#166a9b">OpenCV Otsu thresholding</link> | <link href="https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html" color="#166a9b">OpenCV morphology</link>',True))
    doc=SimpleDocTemplate(str(ROOT/'Module_4_Report.pdf'),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=42,bottomMargin=54,
                          title='Module 4: Thermal animal boundaries',author='Aya Abdelkarem')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print('Created',ROOT/'Module_4_Report.pdf')


if __name__=='__main__':main()
