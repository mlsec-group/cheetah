#!/usr/bin/env bash

set -eu

if [ $# -ne 5 ]; then
    echo "Usage: $0 <memory> <cores> <script_file> <in_args> <out_args>"
    exit 1
fi

memory=$1; shift
cores=$1; shift
script_file=$1; shift
in_args=$1; shift
out_args=$1; shift

echo -e "main($in_args, $out_args)" | joern -J-Xmx$memory -J-XX:ActiveProcessorCount=$cores --import $script_file
