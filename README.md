# SP-NeRF: Integrating Semantic Priors to Enhance Reconstruction from Sparse Satellite Inputs in Weakly Textured Areas (IGARSS 2025)

# 

## Environment Setup

There are two shell scripts provided for creating separate conda environments:

1. **`setup_ba_env.sh`**: Creates a conda environment (e.g., named `ba`) for bundle adjustment and MicMac-related tasks.  
2. **`setup_spnerf_env.sh`**: Creates a conda environment (e.g., named `spnerf`) for training and inference with PyTorch.



After running the scripts, activate the environments using:

```bash
conda activate ba
# or
conda activate spnerf
```



## Data Preparation

The project already includes the converted dataset.

If you wish to **manually construct additional data**, follow the steps below.

> **Note**: Your custom dataset structure must strictly follow the format of the provided dataset to ensure compatibility.



### Refine RPC with Bundle Adjustment

Activate the `ba` environment:

```bash
conda activate ba
```

Run the following command:

```bash
ProjDir=/path/to/your/project
aoi_id=JAX_269
dataset_dir="$ProjDir"/dataset/DFC2019
OutputDir="$ProjDir"/output_dataset/

python -m datasets.create_DFC2019_dataset_auto.py \
    --aoi_id "$aoi_id" \
    --dataset_dir "$dataset_dir" \
    --output_dir "$OutputDir"
```

This refines the RPC parameters using bundle adjustment. Adjust `ProjDir`, `aoi_id`, and other paths according to your setup.



### Generate Dense Depth with MicMac

#### Step 1: Format Conversion

If your original TIFF images use unsupported compression formats (e.g. LZW, Deflate), convert them to a MicMac-compatible format:

```bash
python -m datasets.convert2MicMac.py
```



#### Step 2: Run MicMac

Below is an example for `JAX_214`. Adjust paths and parameters as needed:

```bash
BaseDir=/path/to/your/project
aoi_id=JAX_214
DataDir="$BaseDir"dataset/DFC2019/
RootDir="$BaseDir"output/JAX_214_3_imgs/
TxtDenseDir="$RootDir"dataset"$aoi_id"/root_dir/crops_rpcs_ba_v2/"$aoi_id"/DenseDepth_ZM4/
MicMacDenseDir="$RootDir"DenseDepth/
CodeDir=/path/to/your/project

mkdir "$MicMacDenseDir"
mkdir "$TxtDenseDir"

# Copy images and refined RPC parameters
for line in `cat "$DataDir"train.txt`
do
    img_name=${line%.*}
    cp "$DataDir"RGB/"$aoi_id"/"$img_name".tif "$MicMacDenseDir""$img_name".tif
    cp "$RootDir"ba_files/rpcs_adj/"$img_name".rpc_adj "$MicMacDenseDir""$img_name".txt
done
cp "$DataDir"WGS84toUTM.xml "$MicMacDenseDir"WGS84toUTM.xml
cd "$MicMacDenseDir"

# Convert RPC to MicMac format
mm3d Convert2GenBundle "(.*).tif" "\$1.txt" RPC-d0-adj ChSys=WGS84toUTM.xml Degre=0

for line in `cat "$DataDir"train.txt`
do
    img_name=${line%.*}
    mm3d Malt GeomImage ".*tif" RPC-d0-adj Master="$img_name".tif SzW=1 Regul=0.05 NbVI=2 ZoomF=4 ResolTerrain=1 EZA=1 DirMEC=MM-"$img_name"/
    mm3d TestLib GeoreferencedDepthMap MM-"$img_name" "$img_name".tif Ori-RPC-d0-adj OutDir="$TxtDenseDir" Mask=1 Scale=4
done

cd "$CodeDir"
# Transform 3D points from UTM to geocentric coordinates
python3 utm_to_geocentric.py --file_dir "$TxtDenseDir"
```

> **Note**: Make sure files like `train.txt` and `WGS84toUTM.xml` exist in their expected locations.



## Visualization

To visualize the input/output depth maps:

```bash
python -m visualization.visualize_depth_in
python -m visualization.visualize_depth_out
```



## Training

Here is an example for training SP-NeRF on area `JAX_214` with 3 semantic classes:

```bash
aoi_id=JAX_214
model=sp-nerf
exp_name="$aoi_id"-dense-class3-factor1-downscale8
project_dir=/home/vhr/CPZL/fml/Scene-NeRF

python main.py --aoi_id "$aoi_id" --model "$model" --exp_name "$exp_name" --gpu_id 0 --project_dir "$project_dir" --img_downscale 1 --max_train_steps 30000 --n_importance 0 --lr 0.0005 --sc_lambda 0.1 --depth --ds_lambda 1 --ds_drop 1 --stdscale 1 --sem --num_sem_classes 3 --s_embedding_factor 1 --sem_downscale 8 --dense_ss --ss_lambda 1 --ss_drop 1 --guidedsample --mapping
```

You can modify the arguments above according to your own experiment setup. See `opt.py` for full argument definitions.



## Testing

To evaluate a trained model on a specific checkpoint:

```bash
project_dir=/path/to/your/project
exp_name=/your/expremient/name
dataset_dir=/path/to/your/dataset

python eval.py --project_dir "$project_dir" --exp_name "$exp_name" --dataset_dir "$dataset_dir" --epoch_number 22
```



## Acknowledgements

Our work is inspired by several excellent open-source projects. We sincerely thank the contributors of the following repositories.

- [Sat-NeRF](https://github.com/centreborelli/satnerf)
- [SpS-NeRF](https://github.com/LulinZhang/SpS-NeRF)