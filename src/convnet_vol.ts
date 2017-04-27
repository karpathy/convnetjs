import * as util from "./convnet_util";

/** Vol is the basic building block of all data in a net.
 * it is essentially just a 3D volume of numbers, with a
 * width (sx), height (sy), and depth (depth).
 * it is used to hold data for all filters, all volumes,
 * all weights, and also stores all gradients w.r.t.
 * the data. c is optionally a value to initialize the volume
 * with. If c is missing, fills the Vol with random numbers.
*/
export class Vol {
    public sx: number;
    public sy: number;
    public depth: number;
    public w: number[] | Float64Array;
    public dw: number[] | Float64Array;

    constructor(sx_or_list: number | number[], sy?: number, depth?: number, c?: number) {
        // this is how you check if a variable is an array. Oh, Javascript :)
        // Object.prototype.toString.call(sx_or_list) === '[object Array]'
        if (Array.isArray(sx_or_list)) {
            // we were given a list in sx, assume 1D volume and fill it up
            const list = sx_or_list;
            this.sx = 1;
            this.sy = 1;
            this.depth = list.length;
            // we have to do the following copy because we want to use
            // fast typed arrays, not an ordinary javascript array
            this.w = util.zeros(this.depth);
            this.dw = util.zeros(this.depth);
            for (let i = 0; i < this.depth; i++) {
                this.w[i] = list[i];
            }
        } else {
            // we were given dimensions of the vol
            const sx = sx_or_list;
            this.sx = sx;
            this.sy = sy;
            this.depth = depth;
            const n = sx * sy * depth;
            this.w = util.zeros(n);
            this.dw = util.zeros(n);
            if (typeof c === 'undefined') {
                // weight normalization is done to equalize the output
                // variance of every neuron, otherwise neurons with a lot
                // of incoming connections have outputs of larger variance
                const scale = Math.sqrt(1.0 / (sx * sy * depth));
                for (let i = 0; i < n; i++) {
                    this.w[i] = util.randn(0.0, scale);
                }
            } else {
                for (let i = 0; i < n; i++) {
                    this.w[i] = c;
                }
            }
        }
    }
    get(x: number, y: number, d: number) {
        const ix = ((this.sx * y) + x) * this.depth + d;
        return this.w[ix];
    }
    set(x: number, y: number, d: number, v: number) {
        const ix = ((this.sx * y) + x) * this.depth + d;
        this.w[ix] = v;
    }
    add(x: number, y: number, d: number, v: number) {
        const ix = ((this.sx * y) + x) * this.depth + d;
        this.w[ix] += v;
    }
    get_grad(x: number, y: number, d: number) {
        const ix = ((this.sx * y) + x) * this.depth + d;
        return this.dw[ix];
    }
    set_grad(x: number, y: number, d: number, v: number) {
        const ix = ((this.sx * y) + x) * this.depth + d;
        this.dw[ix] = v;
    }
    add_grad(x: number, y: number, d: number, v: number) {
        const ix = ((this.sx * y) + x) * this.depth + d;
        this.dw[ix] += v;
    }
    cloneAndZero() { return new Vol(this.sx, this.sy, this.depth, 0.0); }
    clone() {
        const V = new Vol(this.sx, this.sy, this.depth, 0.0);
        const n = this.w.length;
        for (let i = 0; i < n; i++) { V.w[i] = this.w[i]; }
        return V;
    }
    addFrom(V: Vol) { for (let k = 0; k < this.w.length; k++) { this.w[k] += V.w[k]; } }
    addFromScaled(V: Vol, a: number) { for (let k = 0; k < this.w.length; k++) { this.w[k] += a * V.w[k]; } }
    setConst(a: number) { for (let k = 0; k < this.w.length; k++) { this.w[k] = a; } }

    toJSON(): VolJSON {
        // todo: we may want to only save d most significant digits to save space
        const json: VolJSON = { sx: this.sx, sy: this.sy, depth: this.depth, w: this.w };
        return json;
        // we wont back up gradients to save space
    }
    fromJSON(json: VolJSON) {
        this.sx = json.sx;
        this.sy = json.sy;
        this.depth = json.depth;

        const n = this.sx * this.sy * this.depth;
        this.w = util.zeros(n);
        this.dw = util.zeros(n);
        // copy over the elements.
        for (let i = 0; i < n; i++) {
            this.w[i] = json.w[i];
        }
    }
}

export interface VolJSON {
    sx: number;
    sy: number;
    depth: number;
    w: number[] | Float64Array;
}
