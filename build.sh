#!/bin/bash

babel src --stage 0 --out-dir lib

browserify index.js > build/convnet.js

cd build

uglifyjs convnet.js --mangle --compress -o convnet.min.js