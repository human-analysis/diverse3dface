1. Create a new python environment by running the following command using conda:
cd diverse3dface
conda env create -f env.yml

2. Download data.zip from the following link:
https://drive.google.com/file/d/1ki8Y3hMy9AL3F_dovLgtViT_dRhdU2fj/view?usp=sharing
Unzip in the diverse3dface directory as diverse3dface/data/

3. Create folder pretrained_models/ inside diverse3dface/.
Download the pretrained models from the following link into the pretrained_models/ folder:
https://drive.google.com/drive/folders/147SVY71dRsNX3sasFq1RIUpGpHP7GuXU?usp=sharing

4. Run the inference code as:
(a) If occlusion-mask available:
python inference.py <path/to/target/image> <path/to/occlusion/mask> <config_file> <output_folder>
Example:
python inference.py examples/1.jpg examples/1-seg_mask.png config_test.config results

(b) If occlusion-mask not available:
python inference.py <path/to/target/image> <config_file> <output_folder>
Example:
python inference.py examples/3.jpg config_test.config results

The output meshes would be generated in the results/<image_name> folder in the .ply format


