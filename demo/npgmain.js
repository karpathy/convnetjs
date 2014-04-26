//Simple game engine
//Author: Andrej Karpathy
//License: BSD
//This function does all the boring canvas stuff. To use it, just create functions:
//update()          gets called every frame
//draw()            gets called every frame
//myinit()          gets called once in beginning
//mouseClick(x, y)  gets called on mouse click
//keyUp(keycode)    gets called when key is released
//keyDown(keycode)  gets called when key is pushed

var canvas;
var ctx;
var WIDTH;
var HEIGHT; 
var FPS;

function drawBubble(x, y, w, h, radius)
{
  var r = x + w;
  var b = y + h;
  ctx.beginPath();
  ctx.strokeStyle="black";
  ctx.lineWidth="2";
  ctx.moveTo(x+radius, y);
  ctx.lineTo(x+radius/2, y-10);
  ctx.lineTo(x+radius * 2, y);
  ctx.lineTo(r-radius, y);
  ctx.quadraticCurveTo(r, y, r, y+radius);
  ctx.lineTo(r, y+h-radius);
  ctx.quadraticCurveTo(r, b, r-radius, b);
  ctx.lineTo(x+radius, b);
  ctx.quadraticCurveTo(x, b, x, b-radius);
  ctx.lineTo(x, y+radius);
  ctx.quadraticCurveTo(x, y, x+radius, y);
  ctx.stroke();
}

function drawRect(x, y, w, h){
  ctx.beginPath();
  ctx.rect(x,y,w,h);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}

function drawCircle(x, y, r){
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI*2, true); 
  ctx.closePath();
  ctx.stroke();
  ctx.fill();
}

//uniform distribution integer
function randi(s, e) {
  return Math.floor(Math.random()*(e-s) + s);
}

//uniform distribution
function randf(s, e) {
  return Math.random()*(e-s) + s;
}

//normal distribution random number
function randn(mean, variance) {
  var V1, V2, S;
  do {
    var U1 = Math.random();
    var U2 = Math.random();
    V1 = 2 * U1 - 1;
    V2 = 2 * U2 - 1;
    S = V1 * V1 + V2 * V2;
  } while (S > 1);
  X = Math.sqrt(-2 * Math.log(S) / S) * V1;
  X = mean + Math.sqrt(variance) * X;
  return X;
}

function eventClick(e) {
    
  //get position of cursor relative to top left of canvas
  var x;
  var y;
  if (e.pageX || e.pageY) { 
    x = e.pageX;
    y = e.pageY;
  } else { 
    x = e.clientX + document.body.scrollLeft + document.documentElement.scrollLeft; 
    y = e.clientY + document.body.scrollTop + document.documentElement.scrollTop; 
  } 
  x -= canvas.offsetLeft;
  y -= canvas.offsetTop;
  
  //call user-defined callback
  mouseClick(x, y, e.shiftKey, e.ctrlKey);
}

//event codes can be found here:
//http://www.aspdotnetfaq.com/Faq/What-is-the-list-of-KeyCodes-for-JavaScript-KeyDown-KeyPress-and-KeyUp-events.aspx
function eventKeyUp(e) {
  var keycode = ('which' in e) ? e.which : e.keyCode;
  keyUp(keycode);
}

function eventKeyDown(e) {
  var keycode = ('which' in e) ? e.which : e.keyCode;
  keyDown(keycode);
}

function NPGinit(FPS){
  //takes frames per secont to run at
  
  canvas = document.getElementById('NPGcanvas');
  ctx = canvas.getContext('2d');
  WIDTH = canvas.width;
  HEIGHT = canvas.height;
  canvas.addEventListener('click', eventClick, false);
  
  //canvas element cannot get focus by default. Requires to either set 
  //tabindex to 1 so that it's focusable, or we need to attach listeners
  //to the document. Here we do the latter
  document.addEventListener('keyup', eventKeyUp, true);
  document.addEventListener('keydown', eventKeyDown, true);
  
  setInterval(NPGtick, 1000/FPS);
  
  myinit();
}

function NPGtick() {
    update();
    draw();
}
