# Thermal photograph provenance

The unaltered photograph saved here as `thermal_dog.jpeg` (original filename
`FLIR_08863.jpeg`) comes from the public
[`sensationTI/FLIR_IR_Expansion` repository](https://github.com/sensationTI/FLIR_IR_Expansion).
Exact source: `misc/FLIR_08863.jpeg` on its `master` branch; Git blob SHA
`3c8bc411f7c1a9b78195f43081e4fc1e78134150`.

The dataset author describes these as FLIR A65 captures and specifically
identifies this photograph as an example containing a dog. The original
640 x 512 image is included without modification. It is a public dataset
photograph, not an image captured by Aya. A copy of the repository's Apache
License 2.0 is included in `IMAGE_LICENSE.txt`.

White-hot polarity was inferred by inspecting the grayscale photograph.
The intensity image does not provide calibrated temperature values.
The dog ROI `[270, 225, 340, 335]` and seed `[301, 284]` were manually
selected. They are disclosed inputs to the classical segmentation.

The resulting boundary is an algorithm estimate. No pixel-accurate reference
mask is supplied here. The repository's bounding-box annotation is not a
segmentation mask, and no accuracy score against it is claimed.
