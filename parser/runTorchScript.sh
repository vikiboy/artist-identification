#!/bin/bash

trainDir1="/home/vikiboy/artistRecognition/data/train_1/"
outputDir1="/home/vikiboy/artistRecognition/code/fast-neural-style/trainOutput/"
scriptPath="/home/vikiboy/artistRecognition/code/fast-neural-style/slow_neural_style.lua"
contentPath="/home/vikiboy/artistRecognition/code/fast-neural-style/1-content.jpg"
extension=".jpg"

for filename in $traindir1;
  do
    name=$(echo $filename | cut -f 1 -d '.')
    outputName=$outputDir1$name$extension
    th slow_neural_style.lua -style_image $filename -content_image $contentPath -style_weights 5.0 -content_weights 0 -gpu 0 -output_image $outputname
  done
