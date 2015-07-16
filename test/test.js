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
		it('should have a field called w', function(){
			var voltype = new convnet.VolType();
			var vol = new voltype();
			assert('w' in vol);
		});
		it('should have a field called dw', function(){
			var voltype = new convnet.VolType();
			var vol = new voltype();
			assert('dw' in vol);
		});
		it('should have a field called sx', function(){
			var voltype = new convnet.VolType();
			var vol = new voltype();
			assert('sx' in vol);
		});
		it('should have a field called sy', function(){
			var voltype = new convnet.VolType();
			var vol = new voltype();
			assert('sy' in vol);
		});
		it('should have a field called depth', function(){
			var voltype = new convnet.VolType();
			var vol = new voltype();
			assert('depth' in vol);
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
			new convnet.ConvLayer({sx:6, filters:4, stride:1, activation:'relu'});
		});
		it('should be able to have a Vol passed through it', function(){
			var voltype = new convnet.VolType(1, 1, 8);
			var vol = new voltype([[[1,2,1,2,1,2,1,2]]]);
			var layer = new convnet.ConvLayer({sx:6, filters:4, stride:1, activation:'relu'});
			layer.forward(vol);
		});
	});
	describe('DropoutLayer', function(){
		it('should work', function(){
			new convnet.DropoutLayer({in_sx: 1, in_sy: 1, in_depth: 2});
		});
		it('should be able to have a Vol passed through it', function(){
			var voltype = new convnet.VolType(1, 1, 2);
			var vol = new voltype([[[1,0]]]);
			var layer = new convnet.DropoutLayer({in_sx: 1, in_sy: 1, in_depth: 2});
			layer.forward(vol);
		});
	});
	describe('FullyConnLayer', function(){
		it('should work', function(){
			new convnet.FullyConnLayer({type:'fc', num_neurons:10, activation:'sigmoid'});
		});
		it('should be able to have a Vol passed through it', function(){
			var voltype = new convnet.VolType(1, 1, 2);
			var vol = new voltype([[[1,0]]]);
			var layer = new convnet.FullyConnLayer({type:'fc', num_neurons:2, activation:'sigmoid'});
			layer.forward(vol);
		});
	});
	describe('InputLayer', function(){
		it('should work', function(){
			new convnet.InputLayer({out_sx:1, out_sy:1, out_depth:20});
		});
		it('should be able to have a Vol passed through it', function(){
			var voltype = new convnet.VolType(1, 1, 2);
			var vol = new voltype([[[1,0]]]);
			var layer = new convnet.InputLayer({in_sx: 1, in_sy: 1, in_depth: 2});
			layer.forward(vol);
		});
	});
	describe('MaxoutLayer', function(){
		it('should work', function(){
			new convnet.MaxoutLayer({group_size: 10, in_sx: 1, in_sy: 1, in_depth: 5});
		});
		it('should be able to have a Vol passed through it', function(){
			var voltype = new convnet.VolType(1, 1, 9);
			var vol = new voltype([[[1,0,2,0,3,0,4,0,5]]]);
			var layer = new convnet.MaxoutLayer({group_size: 3, in_sx: 1, in_sy: 1, in_depth: 9});
			layer.forward(vol);
		});
	});
	describe('PoolLayer', function(){
		it('should work', function(){
			new convnet.PoolLayer({sx: 5, in_sx: 1, in_sy: 1, in_depth: 10});
		});
		it('should be able to have a Vol passed through it', function(){
			var voltype = new convnet.VolType(1, 1, 2);
			var vol = new voltype([[[1,0]]]);
			var layer = new convnet.PoolLayer({sx: 2, in_sx: 1, in_sy: 1, in_depth: 2});
			layer.forward(vol);
		});
	});
	describe('RegressionLayer', function(){
		it('should work', function(){
			new convnet.RegressionLayer({in_sx: 1, in_sy: 1, in_depth: 5});
		});
		it('should be able to have a Vol passed through it', function(){
			var voltype = new convnet.VolType(1, 1, 2);
			var vol = new voltype([[[1,0]]]);
			var layer = new convnet.RegressionLayer({in_sx: 1, in_sy: 1, in_depth: 2});
			layer.forward(vol);
		});
	});
	describe('ReluLayer', function(){
		it('should work', function(){
			new convnet.ReluLayer({in_sx: 1, in_sy: 1, in_depth: 5});
		});
		it('should be able to have a Vol passed through it', function(){
			var voltype = new convnet.VolType(1, 1, 2);
			var vol = new voltype([[[1,0]]]);
			var layer = new convnet.ReluLayer({in_sx: 1, in_sy: 1, in_depth: 2});
			layer.forward(vol);
		});
	});
	describe('SigmoidLayer', function(){
		it('should work', function(){
			new convnet.SigmoidLayer({in_sx: 1, in_sy: 1, in_depth: 5});
		});
		it('should be able to have a Vol passed through it', function(){
			var voltype = new convnet.VolType(1, 1, 2);
			var vol = new voltype([[[1,0]]]);
			var layer = new convnet.SigmoidLayer({in_sx: 1, in_sy: 1, in_depth: 2});
			layer.forward(vol);
		});
	});
	describe('SoftmaxLayer', function(){
		it('should work', function(){
			new convnet.SoftmaxLayer({in_sx: 1, in_sy: 1, in_depth: 5});
		});
		it('should be able to have a Vol passed through it', function(){
			var voltype = new convnet.VolType(1, 1, 2);
			var vol = new voltype([[[1,0]]]);
			var layer = new convnet.SoftmaxLayer({in_sx: 1, in_sy: 1, in_depth: 2});
			layer.forward(vol);
		});
	});
	describe('SVMLayer', function(){
		it('should work', function(){
			new convnet.SVMLayer({in_sx: 1, in_sy: 1, in_depth: 5});
		});
		it('should be able to have a Vol passed through it', function(){
			var voltype = new convnet.VolType(1, 1, 2);
			var vol = new voltype([[[1,0]]]);
			var layer = new convnet.SVMLayer({in_sx: 1, in_sy: 1, in_depth: 2});
			layer.forward(vol);
		});
	});
	describe('TanhLayer', function(){
		it('should work', function(){
			new convnet.TanhLayer({in_sx: 1, in_sy: 1, in_depth: 5});
		});
		it('should be able to have a Vol passed through it', function(){
			var voltype = new convnet.VolType(1, 1, 2);
			var vol = new voltype([[[1,0]]]);
			var layer = new convnet.TanhLayer({in_sx: 1, in_sy: 1, in_depth: 2});
			layer.forward(vol);
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
		it('should be able to have a Vol passed through it', function(){
			var layers = [
			    {type:'input', out_sx:1, out_sy:1, out_depth:2},
			    {type:'fc', num_neurons:20, activation:'relu'},
			    {type:'softmax', num_classes:10}
			];

			var net = new convnet.Net(layers);
			var voltype = new convnet.VolType(1, 1, 2);
			var vol = new voltype([[[1,0]]]);

			net.forward(vol);
		});
	});
});