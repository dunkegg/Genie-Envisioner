export DATASETNAM=habitat_lerobot

python scripts/get_statistics.py \
    --data_root lerobot_data/habitat_lerobot/data/chunk-000 \
    --data_name $DATASETNAM \
    --data_type joint \
    --action_key actions \
    --state_key observation.state \
    --save_path lerobot_data/habitat_lerobot/statistics.json