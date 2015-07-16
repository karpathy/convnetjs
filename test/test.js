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
			assert((new convnet.VolType()) instanceof StructType);
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
			new convnet.ConvLayer({sx:5, filters:8, stride:1, activation:'relu'});
		});
	});
	describe('DropoutLayer', function(){
		it('should work', function(){
			new convnet.DropoutLayer();
		});
	});
	describe('FullyConnLayer', function(){
		it('should work', function(){
			new convnet.FullyConnLayer();
		});
	});
	describe('InputLayer', function(){
		it('should work', function(){
			new convnet.InputLayer({out_sx:1, out_sy:1, out_depth:20});
		});
	});
	describe('MaxoutLayer', function(){
		it('should work', function(){
			new convnet.MaxoutLayer();
		});
	});
});

describe('Nets', function(){
	describe('Net', function(){
		it('should create a network from an array of layers', function(){
			var layers = [
			    {type:'input', out_sx:1, out_sy:1, out_depth:2},
			    {type:'fc', num_neurons:20, activation:'relu'},
			    {type:'softmax', num_classes:10}
			];

			var net = new convnet.Net(layers);
		});
	});
});