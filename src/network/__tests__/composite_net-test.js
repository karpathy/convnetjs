"use strict";
jest.dontMock('../../../build/convnetjs');
jest.dontMock('../composite_net.js');
jest.dontMock('../prime_net.js');
jest.dontMock('../workflow.js');
jest.dontMock('events');

var PrimeNet = require('../prime_net.js');
var CompositeNet = require('../composite_net.js');
var ConvNetJs = require('../../../build/convnetjs');

describe('composite_net', function() {
 it('Test Composite Net Constructor', function() {
   var TestCompositeNet = new CompositeNet();
   expect(TestCompositeNet.networkMap).toBeDefined();
 });
 
 it('Test Composite Net RegisterNetwork', function() {
   var TestCompositeNet = new CompositeNet();
   var TestAutoEncoder = new PrimeNet.AutoEncoder({in_depth: 10});
   var id = 'autoencoder_1';
   
   expect(TestCompositeNet.registerNetwork(TestAutoEncoder, id)).toBe(null);
 });
 
 it('Test Composite Net registerDependentNetwork', function() {
   var TestCompositeNet = new CompositeNet();
   var TestAutoEncoder = new PrimeNet.AutoEncoder({in_depth: 10});
   var TestAutoEncoderDependent = new PrimeNet.AutoEncoder({in_depth: 8});
   
   var id1 = 'autoencoder_1';
   var id2 = 'autoencoder_1_child';
   
   expect(TestCompositeNet.registerNetwork(TestAutoEncoder, id1)).toBe(null);
   expect(TestCompositeNet.registerDependentNetwork(TestAutoEncoderDependent, id2, id1, 0)).toBe(null);
 });
 
 
  it('Test Composite Net fire network chain', function() {
   var TestCompositeNet = new CompositeNet();
   var TestAutoEncoder = new PrimeNet.AutoEncoder({in_depth: 10});
   var TestAutoEncoderDependent = new PrimeNet.AutoEncoder({in_depth: 8});
   
   var id1 = 'autoencoder_1';
   var id2 = 'autoencoder_1_child';
   
   var callback1 = jest.genMockFunction();
   var callback2 = jest.genMockFunction();
   
   TestAutoEncoder.workflow.addFireListener(callback1);
   TestAutoEncoderDependent.workflow.addFireListener(callback2);
   
   //register network
   expect(TestCompositeNet.registerNetwork(TestAutoEncoder, id1)).toBe(null);
   expect(TestCompositeNet.registerDependentNetwork(TestAutoEncoderDependent, id2, id1, 0)).toBe(null);
 
   //fire network
   var inputVol = new ConvNetJs.Vol(1,1,4,0);
   TestCompositeNet.fireNetwork(id1, false, inputVol);
   
   expect(callback1.mock.calls.length).toBe(1);
   expect(callback1).toBeCalledWith(false);
   expect(callback2.mock.calls.length).toBe(1);
   expect(callback2).toBeCalledWith(false);
 });
});


// var CompositeNet = require('../composite_net.js');
// var PrimeNet = require('../prime_net.js');
// // var assign = require('object-assign');

// var TestNet = new CompositeNet();
// var TestAutoEncoder = new PrimeNet.AutoEncoder();




