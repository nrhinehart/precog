export PRECOGROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CURDIR=$(pwd)

# Enable importing from source package.
export PYTHONPATH=$PRECOGROOT:$PYTHONPATH;
export PATH=$PRECOGROOT/scripts:$PATH
