#!/usr/bin/env bash
# Prepare the NVIDIA DRM module for a display variant without touching services.
set -o pipefail

usage() {
    printf 'Usage: bash scripts/display_drm.sh {on|off|check}\n'
    printf '  on    reload nvidia_drm with modeset=1 fbdev=0 after safety checks\n'
    printf '  off   reload nvidia_drm with its default parameters after safety checks\n'
    printf '  check report NVIDIA DRM cards, connector state, and module parameters\n'
}

SYSFS_ROOT=${TF_DISPLAY_DRM_SYSFS:-/sys}
PROC_ROOT=${TF_DISPLAY_DRM_PROC_ROOT:-/proc}
MODULES_ROOT=${TF_DISPLAY_DRM_MODULES:-"$SYSFS_ROOT/module"}
ACTION=${1:-}

read_value() {
    local value
    [[ -r $1 ]] || return 1
    IFS= read -r value < "$1" || return 1
    printf '%s\n' "$value"
}

is_nvidia_device() {
    local device=$1 vendor line
    vendor=$(read_value "$device/vendor" 2>/dev/null || true)
    [[ $vendor == 0x10de ]] && return 0
    [[ -r $device/uevent ]] || return 1
    while IFS= read -r line || [[ -n $line ]]; do
        [[ $line == DRIVER=nvidia ]] && return 0
    done < "$device/uevent"
    return 1
}

discover_devices() {
    local path node
    DISCOVERED_CARDS=()
    NVIDIA_DRM_NODES=()
    for path in "$SYSFS_ROOT"/class/drm/card[0-9]* "$SYSFS_ROOT"/class/drm/renderD[0-9]*; do
        [[ -d $path ]] || continue
        node=${path##*/}
        [[ $node =~ ^(card|renderD)[0-9]+$ ]] || continue
        if is_nvidia_device "$path/device"; then
            NVIDIA_DRM_NODES+=("/dev/dri/$node")
            [[ $node == card* ]] && DISCOVERED_CARDS+=("$node")
        fi
    done
}

scan_outputs() {
    local card=$1 path status enabled
    OUTPUT_STATE=inactive
    ACTIVE_OUTPUTS=()
    for path in "$SYSFS_ROOT"/class/drm/"$card"-*/status; do
        [[ -e $path ]] || continue
        status=$(read_value "$path" 2>/dev/null) || {
            OUTPUT_STATE=unknown
            continue
        }
        [[ $status == connected ]] || continue
        enabled=$(read_value "${path%/status}/enabled" 2>/dev/null) || {
            OUTPUT_STATE=unknown
            continue
        }
        if [[ $enabled == enabled ]]; then
            ACTIVE_OUTPUTS+=("${path%/status}")
            OUTPUT_STATE=active
        elif [[ $enabled != disabled ]]; then
            OUTPUT_STATE=unknown
        fi
    done
}

compute_state() {
    local output pid
    output=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) || return 2
    while IFS= read -r pid; do
        [[ -z ${pid//[[:space:]]/} ]] || return 1
    done <<< "$output"
    return 0
}

is_nvidia_drm_path() {
    local value=$1 node
    for node in "${NVIDIA_DRM_NODES[@]}"; do
        [[ $value =~ ${node}([^[:alnum:]_]|$) ]] && return 0
    done
    return 1
}

mapped_device_state() {
    local maps line proc
    MAP_REASON=
    for maps in "$PROC_ROOT"/[0-9]*/maps; do
        [[ -e $maps ]] || continue
        proc=${maps%/maps}
        proc=${proc##*/}
        [[ -r $maps ]] || {
            MAP_REASON="cannot read maps for pid $proc"
            return 2
        }
        while IFS= read -r line || [[ -n $line ]]; do
            if [[ $line == *libdisplay_kv* ]]; then
                MAP_REASON="pid $proc maps libdisplay_kv"
                return 1
            fi
            if is_nvidia_drm_path "$line"; then
                MAP_REASON="pid $proc maps an NVIDIA DRM device"
                return 1
            fi
        done < "$maps"
    done
    return 0
}

fd_device_state() {
    local procdir fd target proc
    FD_REASON=
    for procdir in "$PROC_ROOT"/[0-9]*; do
        [[ -d $procdir ]] || continue
        proc=${procdir##*/}
        [[ -e $procdir/fd ]] || continue
        [[ -r $procdir/fd ]] || {
            FD_REASON="cannot read fd directory for pid $proc"
            return 2
        }
        for fd in "$procdir"/fd/*; do
            [[ -e $fd || -L $fd ]] || continue
            target=$(readlink "$fd") || {
                FD_REASON="cannot read fd for pid $proc"
                return 2
            }
            if is_nvidia_drm_path "$target"; then
                FD_REASON="pid $proc has an NVIDIA DRM device open"
                return 1
            fi
        done
    done
    return 0
}

module_refcount_state() {
    local refcount
    REFCOUNT_REASON=
    refcount=$(read_value "$MODULES_ROOT/nvidia_drm/refcnt" 2>/dev/null) || return 0
    if [[ $refcount =~ ^[0-9]+$ ]] && ((10#$refcount > 0)); then
        REFCOUNT_REASON="nvidia_drm module refcount is $refcount"
        return 1
    fi
    return 0
}

safe_to_reload() {
    local card state
    compute_state
    case $? in
        0) ;;
        1) printf 'refusing: NVIDIA compute process is running\n' >&2; return 1 ;;
        *) printf 'refusing: cannot determine NVIDIA compute process state\n' >&2; return 1 ;;
    esac
    discover_devices
    ((${#DISCOVERED_CARDS[@]})) || {
        printf 'refusing: no NVIDIA DRM card was found\n' >&2
        return 1
    }
    module_refcount_state
    if [[ $? == 1 ]]; then
        printf 'refusing: %s\n' "$REFCOUNT_REASON" >&2
        return 1
    fi
    mapped_device_state
    case $? in
        0) ;;
        1) printf 'refusing: %s\n' "$MAP_REASON" >&2; return 1 ;;
        *) printf 'refusing: cannot safely inspect DRM mappings: %s\n' "$MAP_REASON" >&2; return 1 ;;
    esac
    fd_device_state
    case $? in
        0) ;;
        1) printf 'refusing: %s\n' "$FD_REASON" >&2; return 1 ;;
        *) printf 'refusing: cannot safely inspect DRM file descriptors: %s\n' "$FD_REASON" >&2; return 1 ;;
    esac
    for card in "${DISCOVERED_CARDS[@]}"; do
        scan_outputs "$card"
        state=$OUTPUT_STATE
        if [[ $state == active ]]; then
            printf 'refusing: active NVIDIA scanout on %s (%s)\n' "$card" "${ACTIVE_OUTPUTS[*]}" >&2
            return 1
        fi
        if [[ $state == unknown ]]; then
            printf 'refusing: connected NVIDIA output state is unknown on %s\n' "$card" >&2
            return 1
        fi
    done
}

restore_default() {
    printf 'attempting to restore default nvidia_drm module settings\n' >&2
    modprobe nvidia_drm >/dev/null 2>&1 || printf 'failed to restore default nvidia_drm module settings\n' >&2
}

reload_module() {
    local mode=$1 attempt
    if ! rmmod nvidia_drm; then
        printf 'failed to unload nvidia_drm\n' >&2
        restore_default
        return 1
    fi
    if [[ $mode == on ]]; then
        modprobe nvidia_drm modeset=1 fbdev=0 || {
            printf 'failed to load nvidia_drm with display parameters\n' >&2
            restore_default
            return 1
        }
    else
        modprobe nvidia_drm || {
            printf 'failed to load nvidia_drm with default parameters\n' >&2
            return 1
        }
    fi
    if [[ $mode == on ]]; then
        for ((attempt = 0; attempt < 5; attempt++)); do
            discover_devices
            ((${#DISCOVERED_CARDS[@]})) && return 0
            sleep 1
        done
        printf 'NVIDIA DRM card did not reappear after display reload\n' >&2
        # The display module is loaded: a plain modprobe would do nothing, so unload it first.
        rmmod nvidia_drm || printf 'failed to unload nvidia_drm before the restore\n' >&2
        restore_default
        return 1
    fi
}

show_check() {
    local card parameter parameter_value
    discover_devices
    if ((${#DISCOVERED_CARDS[@]} == 0)); then
        printf 'nvidia_drm_cards=none\n'
    fi
    for card in "${DISCOVERED_CARDS[@]}"; do
        printf 'card=%s\n' "$card"
        scan_outputs "$card"
        printf 'outputs=%s\n' "$OUTPUT_STATE"
        ((${#ACTIVE_OUTPUTS[@]})) && printf 'active_outputs=%s\n' "${ACTIVE_OUTPUTS[*]}"
    done
    for parameter in modeset fbdev; do
        if parameter_value=$(read_value "$MODULES_ROOT/nvidia_drm/parameters/$parameter" 2>/dev/null); then
            printf '%s=%s\n' "$parameter" "$parameter_value"
        else
            printf '%s=unknown\n' "$parameter"
        fi
    done
    if parameter_value=$(read_value "$MODULES_ROOT/nvidia_drm/refcnt" 2>/dev/null); then
        printf 'nvidia_drm_refcnt=%s\n' "$parameter_value"
    else
        printf 'nvidia_drm_refcnt=unknown\n'
    fi
    if compute_state; then
        printf 'compute=none\n'
    else
        printf 'compute=unknown-or-active\n'
    fi
}

case $ACTION in
    --help|-h|help) usage; exit 0 ;;
    check) show_check ;;
    on|off)
        if [[ $(id -u) != 0 ]]; then
            printf 'on/off require root; run sudo bash scripts/display_drm.sh %s\n' "$ACTION" >&2
            exit 1
        fi
        safe_to_reload || exit 1
        reload_module "$ACTION"
        ;;
    *) usage >&2; exit 2 ;;
esac
