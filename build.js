#!/usr/bin/env node

var fs = require('fs');
var path = require('path');
var babelify = require('babelify');
var browserify = require('browserify');
var uglify = require('uglify-js');

var b = browserify();
b.add("./index.js");
b.transform(babelify.configure({
	stage : 0
}));
b.bundle(function(error, buffer){
	if(error){
		console.error(error);
	}
	fs.writeFileSync(
		path.join(process.cwd(), "./build/convnet.js"), 
		buffer.toString('utf8')
	);
	fs.writeFileSync(
		path.join(process.cwd(), "./build/convnet.min.js"), 
		uglify.minify(
			buffer.toString('utf8'), 
			{
				fromString: true
			}
		).code
	);
});