#!/usr/bin/env bash
# find_family.sh - Auto-detect the config subdirectory (family) for a given model name.
# Usage: source submit/find_family.sh <model_name>
#   Sets FAMILY variable to the config subdirectory containing <model_name>.yaml
#   Exits with error if not found or if duplicates exist.

_model_name="$1"
if [ -z "$_model_name" ]; then
    echo "Error: No model name provided." >&2
    echo "Usage: source submit/find_family.sh <model_name>" >&2
    exit 1
fi

_configs_dir="${WORKSPACE_PATH:-$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")}/configs"
_matches=()

for _yaml in "$_configs_dir"/*/"${_model_name}.yaml"; do
    if [ -f "$_yaml" ]; then
        _dir=$(basename "$(dirname "$_yaml")")
        _matches+=("$_dir")
    fi
done

if [ ${#_matches[@]} -eq 0 ]; then
    echo "Error: No config found for model '${_model_name}' in ${_configs_dir}/*/" >&2
    exit 1
elif [ ${#_matches[@]} -gt 1 ]; then
    echo "Warning: Model '${_model_name}' found in multiple families: ${_matches[*]}" >&2
    echo "Error: Ambiguous model name. Please specify the family manually." >&2
    exit 1
fi

FAMILY="${_matches[0]}"
echo "Auto-detected family: ${FAMILY}"
