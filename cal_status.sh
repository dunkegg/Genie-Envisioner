export DATASETNAM=move_to_object

python scripts/get_statistics.py \
    --data_root lerobot_data/move_to_object/data/chunk-000 \
    --data_name $DATASETNAM \
    --data_type joint \
    --action_key actions \
    --state_key observation.state \
    --save_path lerobot_data/move_to_object/status.json