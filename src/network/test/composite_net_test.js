"use strict";

var CompositeNet = require('../composite_net.js');
var PrimeNet = require('../prime_net.js');
// var assign = require('object-assign');

var TestNet = new CompositeNet();
var TestAutoEncoder = new PrimeNet.AutoEncoder();


// //Exports
module.exports.TestNet = CompositeNet;

