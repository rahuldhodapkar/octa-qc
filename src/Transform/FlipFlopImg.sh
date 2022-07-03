#!/usr/bin/env bash
#
## FlipFlopImg.sh
# Create copies of original images with horizontal and vertical
# mirroring to increase training and test set size.
#
# @author rahuldhodapkar <rahul.dhodapkar@yale.edu>
# @version 2021.05.18
#

set -e
set -x

mkdir -p calc/transform/vertflip
mkdir -p calc/transform/horzflop
mkdir -p calc/transform/flipflop

for f in `ls data/images`
do
    convert -flip data/images/$f calc/transform/vertflip/$f
    convert -flop data/images/$f calc/transform/horzflop/$f
    convert -flip -flop data/images/$f calc/transform/flipflop/$f
done

echo "All done!"