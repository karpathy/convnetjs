#!/bin/bash

babel src --stage 0 --out-dir lib

browserify index.js > ./build/convnet.js

uglifyjs --mangle --compress ./build/convnet.js -o ./build/convnet.min.js