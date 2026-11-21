# Module 4 requirement mapping

| Assignment requirement | Included evidence |
|---|---|
| Thermal-camera image containing an animal | Unmodified public FLIR A65 dog image, source attribution and license |
| Implement boundary extraction without ML/deep learning | Independent OpenCV Gaussian/Otsu/morphology/component/contour branch; no neural inference in the primary method |
| Return animal boundary | Full-resolution mask, green outline and 247 ordered x,y boundary vertices |
| Compare with SAM2 segmentation | Actual official SAM2.1 Tiny CPU run, preserved mask/logits, matched image/box, IoU/Dice and boundary-distance comparison |
| Accessible assignment-specific GitHub repository | Repository link on report page 1 |
| Commented code and execution documentation at each script's top | Header docstrings in all four Python scripts and reproduction commands in README |
| Working application demonstration | Captioned MP4 from a fresh classical and SAM2 execution; source and decoded-frame validation |
| PDF and worked examples | Four-page PDF with numerical overlap calculation and visual comparisons |
| Google Classroom PDF/video submission | Files prepared; final attachment and turn-in left to Aya |

The instruction requests exact boundaries. This implementation returns the exact external contour of its estimated binary mask, but no pixel-accurate reference establishes that the estimate is the exact true silhouette. The report explicitly discusses visible missed details. Comparison against SAM2 is not proof of ground-truth accuracy.

The recording is actual application-rendered output, not an OS desktop capture. The assignment allows a screen recording or properly captured footage of the working system; the recording method is explicitly disclosed.
