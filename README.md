An AI model to predict tumor hypoxia (FMISO TBR) from FDG scan.

## Reference
Predicting F-18 FMISO PET Hypoxia Measurements From F-18 FDG PET Scan Using a Generative Adversarial Network. Wei Zhao, Milan Grkovski, Nancy Lee, Heiko Schoder, John Humm, Harini Veeraraghavan, Joseph O. Deasy, Memorial Sloan Kettering Cancer Center, New York, NY, Presented at AAPM 2022 Annual meeting PO-GePV-I-82 (Sunday, 7/10/2022).

## Installation
Clone this repository and run the following command
```
conda env create -f environment.yml
```

## Running Inference
```
python infer_tbr.py /path/to/fdg_and_gtv_seg /path/to/TBR_output
```
The FDG image should be in lean body mass SUV. The inference script includes the following pre-processing steps:
- Image resampling: Bilinear interpolation for FDG, nearest neighbor for tumor mask
- Zero-out pixel values outside the tumor mask, split tumor segmentation into connected components and extract 32 by 32 axial slice where tumor is centered
- Normalization of image pixel values to [-1, 1]

For user’s reference, FDG scan input and the expected model output for one phantom dataset are provided in phantom folder.

## License

By downloading the software you are agreeing to the following terms and conditions as well as to the Terms of Use.

THE SOFTWARE IS PROVIDED "AS IS" AND Service for Predictive Informatics at MSKCC AND ITS COLLABORATORS DO NOT MAKE ANY WARRANTY, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, NOR DO THEY ASSUME ANY LIABILITY OR RESPONSIBILITY FOR THE USE OF THIS SOFTWARE.

This software is for research purposes only and has not been approved for clinical or commercial use.

Software has not been reviewed or approved by the Food and Drug Administration, and is for non-clinical, non-commercial, IRB-approved Research Use Only. In no event shall data or images generated through the use of the Software be used in the provision of patient care.

Memorial Sloan Kettering Cancer Center (MSKCC), New York NY retains all rights to commercialize and license this software.

YOU MAY NOT DISTRIBUTE COPIES of this software, or copies of software derived from this software, to others outside your organization without specific prior written permission from the Service for Predictive Informatics, Medical Physics at MSKCC except where noted for specific software products.
All Technology and technical data delivered under this Agreement are subject to US export control laws and may be subject to export or import regulations in other countries. You agree to comply strictly with all such laws and regulations and acknowledge that you have the responsibility to obtain such licenses to export, re-export, or import as may be required after delivery to you.

You may publish papers and books using results produced using software provided you cite the following:
<br>AI models: https://doi.org/10.1088/1361-6560/ac4000 https://doi.org/10.48550/arXiv.1909.05054
<br>CERR model library: https://doi.org/10.1016/j.ejmp.2020.04.011

