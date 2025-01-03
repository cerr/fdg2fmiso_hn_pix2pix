AI model to convert FDG scan to TBR.

## Reference
Predicting F-18 FMISO PET Hypoxia Measurements From F-18 FDG PET Scan Using a Generative Adversarial Network. Wei Zhao, Milan Grkovski, Nancy Lee, Heiko Schoder, John Humm, Harini Veeraraghavan, Joseph O. Deasy, Memorial Sloan Kettering Cancer Center, New York, NY, Presented at AAPM 2022 Annual meeting PO-GePV-I-82 (Sunday, 7/10/2022).

## Installation
Clone this repository and run the following command
conda env create -f environment.yml

## Running Inference
Place NifTi files for FDG scan and GTV segmentation in a directory. The FDG scan file must be named as fdg_scan.nii.gz and GTV segmentationfile must be named as gtv_seg.nii.gz. Then, run inference as follows:

python infer_tbr.py /path/to/directory/containnig/fdg_and_gtv_seg /path/to/TBR_output_directory
