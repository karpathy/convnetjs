// a window stores _size_ number of values
// and returns averages. Useful for keeping running
// track of validation or training accuracy during SGD

export default class Window {

  constructor(size = 100, minsize = 20){
    this.v = [];
    this.size = typeof(size)==='undefined' ? 100 : size;
    this.minsize = typeof(minsize)==='undefined' ? 20 : minsize;
    this.sum = 0;
  }

  add(x) {
    this.v.push(x);
    this.sum += x;
    if(this.v.length>this.size) {
      this.sum -= this.v.shift();;
    }
  }

  getAverage() {
    if(this.v.length < this.minsize){
      return -1;
    } else {
      return this.sum/this.v.length;
    }
  }

  reset(x) {
    this.v = [];
    this.sum = 0;
  }

}