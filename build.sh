#!/bin/bash

babel src --out-dir lib

browserify index.js > ./build/convnet.js

uglify --mangle --compress ./build/convnet.js -o ./build/convnet.min.js