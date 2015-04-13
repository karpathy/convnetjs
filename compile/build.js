#!/usr/bin/env node

var fs = require('fs');
var path = require('path');
var babelify = require('babelify');
var browserify = require('browserify');
var uglify = require('uglify-js');

var b = browserify();
b.add("../src/convnet_export.js");
b.transform(babelify);
b.bundle(function(error, buffer){
	fs.writeFileSync(
		path.join(process.cwd(), "../build/convnet.js"), 
		buffer.toString('utf8')
	);
	fs.writeFileSync(
		path.join(process.cwd(), "../build/convnet-min.js"), 
		UglifyJS.minify(
			buffer.toString('utf8'), 
			{
				fromString: true
			}
		)
	);
});