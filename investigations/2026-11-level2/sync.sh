#!/bin/sh
#  copies the source tree to /tmp/l2study/<dir> on an E44 host (two hops
#  through theon):  sync.sh HOST DIR
set -e
host=$1
dir=$2
top=$(cd "$(dirname "$0")/../.." && pwd)
tmp=/tmp/l2study-lehn-$$.tar.gz
COPYFILE_DISABLE=1 tar -C "$top" --exclude .git --exclude 'build*' \
    --exclude '*.o' --exclude '*.a' --exclude '*.so' --exclude '*.dylib' \
    -czf "$tmp" .
scp -q "$tmp" theon:"$tmp"
ssh theon "scp -q $tmp $host:$tmp && rm $tmp && ssh $host 'mkdir -p /tmp/l2study/$dir && tar -C /tmp/l2study/$dir -xzf $tmp 2>/dev/null; rm $tmp'"
rm "$tmp"
