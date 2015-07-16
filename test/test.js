var assert = require('assert');
var mocha = require('mocha');

var convnet = require('../index.js');

var it = mocha.it;
var describe = mocha.describe;

describe('Structures', function(){
	describe('VolType', function(){
		it('should make a new VolType', function(){
			new convnet.VolType(1, 1, 2);
		});
		it('should be an instance of StructType', function(){
			assert((new VolType()) instanceof StructType);
		});
	});
	describe('Experience', function(){
		it('should work', function(){
			new convnet.Experience();
		});
	});
	describe('Window', function(){
		it('should work', function(){
			new convnet.Window();
		});
	});
});

describe('Layers', function(){
	describe('ConvLayer', function(){
		it('should work', function(){
			new convnet.ConvLayer();
		});
	});
	describe('DropoutLayer', function(){
		it('should work', function(){

		});
	});
});

describe('Nets', function(){
	it('should create a network from an array of layers', function(){
		var layers = [
		    {type:'input', out_sx:1, out_sy:1, out_depth:2},
		    {type:'fc', num_neurons:20, activation:'relu'},
		    {type:'softmax', num_classes:10}
		];

		var net = new convnet.Net(layers);
	});
});