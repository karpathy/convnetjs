"use strict";
jest.dontMock('../../../build/convnetjs');
jest.dontMock('../prime_net.js');
jest.dontMock('../workflow.js');
jest.dontMock('events');

var PrimeNet = require('../prime_net.js');
var ConvNetJs = require('../../../build/convnetjs');

describe('prime_net test', function() {
 it('Test AutoEncoder Constructor, in_depth: 10', function() {
   var TestAutoEncoder = new PrimeNet.AutoEncoder({in_depth: 10});
   expect(TestAutoEncoder.inputVol.getLength()).toEqual(10);
 });
 
 it('Test AutoEncoder Fire', function() {
   var inputVol = new ConvNetJs.Vol(1,1,4,0);
   var TestAutoEncoder = new PrimeNet.AutoEncoder({in_depth: 10});
   
   var callback = jest.genMockFunction();
   
   TestAutoEncoder.workflow.addFireListener(callback);
   
   expect(TestAutoEncoder.forward(false, inputVol)).toEqual(TestAutoEncoder.act);
   expect(callback.mock.calls.length).toBe(1);
   expect(callback).toBeCalledWith(false);
 });
 
});



