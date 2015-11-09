"use strict";
jest.dontMock('../../../build/convnetjs');
jest.dontMock('../util.js');

var convnetjs = require('../../../build/convnetjs');
var smartCopy = require('../util').smartCopy;

describe('util-smart-copy', function() {
 it('copy v1 [0,0] to v2 [1,1,1,1] at location 0', function() {
   
   var v1 = new convnetjs.Vol(1,1,2,0);
   var v2 = new convnetjs.Vol(1,1,4,1);
   smartCopy(v1, v2, 0);
   
   expect(v2.getW().toString()).toBe([0,0,1,1].toString());
 });
 
 it('copy v1 [0,0] to v2 [1,1,1,1] at location 2', function() {
   
   var v1 = new convnetjs.Vol(1,1,2,0);
   var v2 = new convnetjs.Vol(1,1,4,1);
   smartCopy(v1, v2, 2);
   
   expect(v2.getW().toString()).toBe([1,1,0,0].toString());
 });
 
 it('copy v1 [0,0] to v2 [1,1,1,1] at location 3', function() {
   
   var v1 = new convnetjs.Vol(1,1,2,0);
   var v2 = new convnetjs.Vol(1,1,4,1);
   smartCopy(v1, v2, 3);
   
   expect(v2.getW().toString()).toBe([1,1,1,0].toString());
 });
});