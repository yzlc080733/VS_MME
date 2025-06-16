# Visual search scanpath modeling

Repository for our work on visual search.

### Dataset preparation

```bash
### COCO-Search18
# Download raw dataset and place it in /data/COCOSearch18_dataset/
python prepare_data_cocosearch18.py

### ASD dataset
# Download raw dataset and place it in ./env_data/
python prepare_data_ASD.py

### IVSN dataset
# Download dataset and place it in ./reference_IVSN_org/
python prepare_data_IVSNP1.py
```

See python codes for sources of the datasets.

The processed data will be stored in `./env_data`. The size can be around 1.5GB.

### Conda environment

```bash
conda env create -f conda_exp.yml
```

### How to run

Example for running the MME method implemented by SVPG.

```bash
python run_exp.py --valsplit 0 --dataset cocosearch18-All --method MEPPO-0-0.0-1.0-10.0-snn-2.0-0.0001-0.00001 --rep 0 --thread 10 --cuda 0
```

See `run_exp.py` for details in arguments.

The results include the following. Some methods do not generate training records and models.

* Predicted agent scanpath: `./log_text/agenttraj_<dataset>_<method>_<repeat_id>.pkl`;

* Training record: `./log_text/log_<dataset>_<method>_<repeat_id>.txt`;

* Model (components): `./log_model/<dataset>_<method>_<repeat_id>_<component>.pt`
