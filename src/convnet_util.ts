// Random number utilities
let return_v: boolean = false;
let v_val: number = 0.0;
export function gaussRandom(): number {
    if (return_v) {
        return_v = false;
        return v_val;
    }
    const u = 2 * Math.random() - 1;
    const v = 2 * Math.random() - 1;
    const r = u * u + v * v;
    if (r === 0 || r > 1) { return gaussRandom(); }
    const c = Math.sqrt(-2 * Math.log(r) / r);
    v_val = v * c; // cache this
    return_v = true;
    return u * c;
}
export function randf(a: number, b: number) { return Math.random() * (b - a) + a; }
export function randi(a: number, b: number) { return Math.floor(Math.random() * (b - a) + a); }
export function randn(mu: number, std: number) { return mu + gaussRandom() * std; }

// Array utilities
export function zeros(n?: number): number[] | Float64Array {
    if (typeof (n) === 'undefined' || isNaN(n)) { return []; }
    if (typeof ArrayBuffer === 'undefined') {
        // lacking browser support
        const arr = new Array(n);
        for (let i = 0; i < n; i++) { arr[i] = 0; }
        return arr;
    } else {
        return new Float64Array(n);
    }
}

export function arrContains<T>(arr: T[], elt: T) {
    for (let i = 0, n = arr.length; i < n; i++) {
        if (arr[i] === elt) { return true; }
    }
    return false;
}

export function arrUnique<T>(arr: T[]) {
    const b = [];
    for (let i = 0, n = arr.length; i < n; i++) {
        if (!arrContains(b, arr[i])) {
            b.push(arr[i]);
        }
    }
    return b;
}

import { MaxMinResult } from "./cnnutil";
/** return max and min of a given non-empty array. */
export function maxmin(w: number[] | Float64Array): MaxMinResult {
    if (w.length === 0) { return {}; } // ... ;s
    let maxv = w[0];
    let minv = w[0];
    let maxi = 0;
    let mini = 0;
    const n = w.length;
    for (let i = 1; i < n; i++) {
        if (w[i] > maxv) { maxv = w[i]; maxi = i; }
        if (w[i] < minv) { minv = w[i]; mini = i; }
    }
    return { maxi: maxi, maxv: maxv, mini: mini, minv: minv, dv: maxv - minv };
}

/** create random permutation of numbers, in range [0...n-1] */
export function randperm(n: number): number[] {
    let i = n,
        j = 0,
        temp;
    const array = [];
    for (let q = 0; q < n; q++) {
        array[q] = q;
    }
    while (i--) {
        j = Math.floor(Math.random() * (i + 1));
        temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
    return array;
}

/** sample from list lst according to probabilities in list probs
 * the two lists are of same size, and probs adds up to 1
 */
export function weightedSample(lst: number[], probs: number[]): number {
    const p = randf(0, 1.0);
    let cumprob = 0.0;
    for (let k = 0, n = lst.length; k < n; k++) {
        cumprob += probs[k];
        if (p < cumprob) { return lst[k]; }
    }
}

/**
 * syntactic sugar function for getting default parameter values
 */
export function getopt(opt: { [key: string]: number | string }, field_name: string | string[], default_value: number): number {
    if (typeof field_name === 'string') {
        // case of single string
        const ret = (typeof opt[field_name] !== 'undefined') ? opt[field_name] : default_value;
        return Number(ret);
    } else {
        // assume we are given a list of string instead
        let ret = default_value;
        for (let i = 0; i < field_name.length; i++) {
            const f = field_name[i];
            if (typeof opt[f] !== 'undefined') {
                ret = Number(opt[f]); // overwrite return value
            }
        }
        return ret;
    }
}

export function assert(condition: boolean, message: string) {
    if (!condition) {
        message = message || "Assertion failed";
        if (typeof Error !== "undefined") {
            throw new Error(message);
        }
        throw message; // Fallback
    }
}

