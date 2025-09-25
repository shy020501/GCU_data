#!/usr/bin/env bash
set -euo pipefail

data_dir="./data"
prompt_path="${data_dir}/prompts"

celebrities=(
  adam_driver
  adriana_lima
  amber_heard
  amy_adams
  andrew_garfield
  angelina_jolie
  anjelica_huston
  anna_kendrick
  bill_gates
  elon_musk
)
modes=(short_adv extended extended_adv)

# Generate available prompts per celebrity
for celebrity in "${celebrities[@]}"; do
  for mode in "${modes[@]}"; do
    echo ">> [$(date '+%F %T')] Processing: ${celebrity} / ${mode}"
    python create_avail_prompt.py \
      --task celebrity \
      --concept "${celebrity}" \
      --mode "${mode}" \
      --output_path "${data_dir}" \
      --prompt_path "${prompt_path}"
  done
done

# Build dataset
echo ">> [$(date '+%F %T')] Building dataset"
python build_dataset.py --task celebrity --base_dir "${data_dir}"

echo ">> Done."
