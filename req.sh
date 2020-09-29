#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd)"
cd $SCRIPTPATH

if [[ $1 == "install" ]]; then
    source env/bin/activate
    set -x
    python -m pip install -r requirements.txt
elif [[ $1 == "freeze" ]]; then
    source env/bin/activate
    set -x
    python -m pip freeze > requirements.txt
else
    echo "Help!"
    echo "   Freeze: ./req.sh freeze"
    echo "  Install: ./req.sh install"
fi
