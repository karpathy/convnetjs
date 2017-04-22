
// contains various utility functions

/**
 * a window stores _size_ number of values
 * and returns averages. Useful for keeping running
 * track of validation or training accuracy during SGD
 */
export class Window {
    public v: number[];
    public size: number;
    public minsize: number;
    public sum: number;
    constructor(size?: number, minsize?: number) {
        this.v = [];
        this.size = typeof (size) === 'undefined' ? 100 : size;
        this.minsize = typeof (minsize) === 'undefined' ? 20 : minsize;
        this.sum = 0;
    }
    add(x: number) {
        this.v.push(x);
        this.sum += x;
        if (this.v.length > this.size) {
            const xold = this.v.shift();
            this.sum -= xold;
        }
    }
    get_average(): number {
        if (this.v.length < this.minsize) { return -1; }
        else { return this.sum / this.v.length; }
    }
    reset() {
        this.v = [];
        this.sum = 0;
    }
}

export interface MaxMinResult{
    maxi?:number;
    maxv?:number;
    mini?:number;
    minv?:number;
    dv?:number;
}

/**
 * returns min, max and indeces of an array
 */
export function maxmin(w: number[]) :MaxMinResult{
    if (w.length === 0) { return {}; } // ... ;s

    let maxv = w[0];
    let minv = w[0];
    let maxi = 0;
    let mini = 0;
    for (let i = 1; i < w.length; i++) {
        if (w[i] > maxv) { maxv = w[i]; maxi = i; }
        if (w[i] < minv) { minv = w[i]; mini = i; }
    }
    return { maxi: maxi, maxv: maxv, mini: mini, minv: minv, dv: maxv - minv };
}

/**
 * returns string representation of float
 * but truncated to length of d digits
 */
export function f2t(x: number, d?: number):string {
    if (typeof (d) === 'undefined') { d = 5; }
    const dd = 1.0 * Math.pow(10, d);
    return '' + Math.floor(x * dd) / dd;
}


