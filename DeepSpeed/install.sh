#!/bin/bash

set -e

# Error handler
err_report() {
    echo "Error on line $1"
    echo "Failed to install DeepSpeed"
}
trap 'err_report $LINENO' ERR

# Usage information
usage() {
  echo """
Usage: install.sh [options...]

Installs DeepSpeed and optionally its third-party dependencies.

[optional]
    -d, --deepspeed_only    Install only DeepSpeed without third-party dependencies
    -t, --third_party_only  Install only third-party dependencies
    -l, --local_only        Install only on the local machine
    -s, --pip_sudo          Run pip with sudo (default: no sudo)
    -m, --pip_mirror        Specify a pip mirror URL (default: the default pip mirror)
    -H, --hostfile          Path to hostfile for multi-node installation (default: /job/hostfile)
    -a, --apex_commit       Install a specific commit hash of apex
    -k, --skip_requirements Skip installing DeepSpeed requirements
    -h, --help              Display this help text
  """
}

# Default options
ds_only=0
tp_only=0
deepspeed_install=1
third_party_install=1
local_only=1
pip_sudo=0
hostfile=/job/hostfile
pip_mirror=""
apex_commit=""
skip_requirements=0

# Parse arguments
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -d|--deepspeed_only)
    deepspeed_install=1
    third_party_install=0
    ds_only=1
    shift
    ;;
    -t|--third_party_only)
    deepspeed_install=0
    third_party_install=1
    tp_only=1
    shift
    ;;
    -l|--local_only)
    local_only=1
    shift
    ;;
    -s|--pip_sudo)
    pip_sudo=1
    shift
    ;;
    -m|--pip_mirror)
    pip_mirror=$2
    shift
    shift
    ;;
    -a|--apex_commit)
    apex_commit=$2
    shift
    shift
    ;;
    -k|--skip_requirements)
    skip_requirements=1
    shift
    ;;
    -H|--hostfile)
    hostfile=$2
    if [ ! -f $hostfile ]; then
        echo "Hostfile does not exist: $hostfile"
        exit 1
    fi
    shift
    shift
    ;;
    -h|--help)
    usage
    exit 0
    ;;
    *)
    echo "Unknown argument: $1"
    usage
    exit 1
    ;;
esac
done

# Ensure mutual exclusivity
if [ "$ds_only" == "1" ] && [ "$tp_only" == "1" ]; then
    echo "-d and -t options are mutually exclusive"
    usage
    exit 1
fi

# Update git version information
echo "Updating git version information"
echo "git_hash = '$(git rev-parse --short HEAD)'" > deepspeed/git_version_info.py
echo "git_branch = '$(git rev-parse --abbrev-ref HEAD)'" >> deepspeed/git_version_info.py

# Configure pip options
if [ "$pip_sudo" == "1" ]; then
  PIP_SUDO="sudo -H"
else
  PIP_SUDO=""
fi

if [ "$pip_mirror" != "" ]; then
  PIP_INSTALL="pip install -i $pip_mirror"
else
  PIP_INSTALL="pip install"
fi

# Install requirements
if [ "$skip_requirements" == "0" ]; then
    $PIP_SUDO $PIP_INSTALL -r requirements.txt
fi

# Build third-party dependencies
if [ "$third_party_install" == "1" ]; then
    echo "Building Apex"
    cd third_party/apex
    if [ "$apex_commit" != "" ]; then
        git fetch
        git checkout $apex_commit
    fi
    python setup.py --cpp_ext --cuda_ext bdist_wheel
    cd -
    $PIP_SUDO pip uninstall -y apex
    $PIP_SUDO $PIP_INSTALL third_party/apex/dist/apex*.whl
fi

# Build DeepSpeed
if [ "$deepspeed_install" == "1" ]; then
    echo "Building DeepSpeed"
    python setup.py bdist_wheel
    $PIP_SUDO pip uninstall -y deepspeed
    $PIP_SUDO $PIP_INSTALL dist/deepspeed*.whl
fi

# Local or multi-node installation
if [ "$local_only" != "1" ]; then
    echo "Installing across nodes"
    local_path=$(pwd)
    tmp_wheel_path="/tmp/deepspeed_wheels"
    pdsh -w $hosts "mkdir -p $tmp_wheel_path"
    pdcp -w $hosts dist/deepspeed*.whl $tmp_wheel_path/
    pdsh -w $hosts "$PIP_SUDO pip uninstall -y deepspeed"
    pdsh -w $hosts "$PIP_SUDO $PIP_INSTALL $tmp_wheel_path/deepspeed*.whl"
    pdsh -w $hosts "rm -rf $tmp_wheel_path"
fi

echo "Installation complete"
