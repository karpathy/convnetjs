
// contains various utility functions 
var cnnvis = (function(exports){

  // can be used to graph loss, or accuract over time
  var Graph = function(options) {
    var options = options || {};
    this.step_horizon = options.step_horizon || 1000;
    
    this.pts = [];
    
    this.maxy = -9999;
    this.miny = 9999;
  }

  Graph.prototype = {
    // canv is the canvas we wish to update with this new datapoint
    add: function(step, y) {
      var time = new Date().getTime(); // in ms
      if(y>this.maxy*0.99) this.maxy = y*1.05;
      if(y<this.miny*1.01) this.miny = y*0.95;

      this.pts.push({step: step, time: time, y: y});
      if(step > this.step_horizon) this.step_horizon *= 2;
    },
    // elt is a canvas we wish to draw into
    drawSelf: function(canv) {
      
      var pad = 25;
      var H = canv.height;
      var W = canv.width;
      var ctx = canv.getContext('2d');

      ctx.clearRect(0, 0, W, H);
      ctx.font="10px Georgia";

      var f2t = function(x) {
        var dd = 1.0 * Math.pow(10, 2);
        return '' + Math.floor(x*dd)/dd;
      }

      // draw guidelines and values
      ctx.strokeStyle = "#999";
      ctx.beginPath();
      var ng = 10;
      for(var i=0;i<=ng;i++) {
        var xpos = i/ng*(W-2*pad)+pad;
        ctx.moveTo(xpos, pad);
        ctx.lineTo(xpos, H-pad);
        ctx.fillText(f2t(i/ng*this.step_horizon/1000)+'k',xpos,H-pad+14);
      }
      for(var i=0;i<=ng;i++) {
        var ypos = i/ng*(H-2*pad)+pad;
        ctx.moveTo(pad, ypos);
        ctx.lineTo(W-pad, ypos);
        ctx.fillText(f2t((ng-i)/ng*(this.maxy-this.miny) + this.miny), 0, ypos);
      }
      ctx.stroke();

      var N = this.pts.length;
      if(N<2) return;

      // draw the actual curve
      var t = function(x, y, s) {
        var tx = x / s.step_horizon * (W-pad*2) + pad;
        var ty = H - ((y-s.miny) / (s.maxy-s.miny) * (H-pad*2) + pad);
        return {tx:tx, ty:ty}
      }

      ctx.strokeStyle = "red";
      ctx.beginPath()
      for(var i=0;i<N;i++) {
        // draw line from i-1 to i
        var p = this.pts[i];
        var pt = t(p.step, p.y, this);
        if(i===0) ctx.moveTo(pt.tx, pt.ty);
        else ctx.lineTo(pt.tx, pt.ty);
      }
      ctx.stroke();
    }
  }

  exports = exports || {};
  exports.Graph = Graph;
  return exports;

})(typeof module != 'undefined' && module.exports);  // add exports to module.exports if in node.js


