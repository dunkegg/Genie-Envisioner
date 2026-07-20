export DATASETNAM=R2R

python scripts/get_statistics.py \
    --data_root lerobot_data/R2R/data \
    --data_name $DATASETNAM \
    --data_type joint \
    --action_key actions \
    --state_key observation.state \
    --save_path lerobot_data/R2R/statistics.json