import { Vol } from "./convnet_vol";
import * as uitl from "./convnet_util";
// Volume utilities
/**
 * intended for use with data augmentation
 * crop is the size of output
 * dx,dy are offset wrt incoming volume, of the shift
 * fliplr is boolean on whether we also want to flip left<->right
*/
export function augment(V: Vol, crop: number, dx?: number, dy?: number, fliplr?: boolean) {
    // note assumes square outputs of size crop x crop
    if (typeof (fliplr) === 'undefined') { fliplr = false; }
    if (typeof (dx) === 'undefined') { dx = uitl.randi(0, V.sx - crop); }
    if (typeof (dy) === 'undefined') { dy = uitl.randi(0, V.sy - crop); }

    // randomly sample a crop in the input volume
    let W: Vol;
    if (crop !== V.sx || dx !== 0 || dy !== 0) {
        W = new Vol(crop, crop, V.depth, 0.0);
        for (let x = 0; x < crop; x++) {
            for (let y = 0; y < crop; y++) {
                if (x + dx < 0 || x + dx >= V.sx || y + dy < 0 || y + dy >= V.sy) { continue; } // oob
                for (let d = 0; d < V.depth; d++) {
                    W.set(x, y, d, V.get(x + dx, y + dy, d)); // copy data over
                }
            }
        }
    } else {
        W = V;
    }

    if (fliplr) {
        // flip volume horziontally
        const W2 = W.cloneAndZero();
        for (let x = 0; x < W.sx; x++) {
            for (let y = 0; y < W.sy; y++) {
                for (let d = 0; d < W.depth; d++) {
                    W2.set(x, y, d, W.get(W.sx - x - 1, y, d)); // copy data over
                }
            }
        }
        W = W2; //swap
    }
    return W;
}

// img is a DOM element that contains a loaded image
// returns a Vol of size (W, H, 4). 4 is for RGBA
export function img_to_vol(img: HTMLImageElement, convert_grayscale?: boolean) {

    if (typeof (convert_grayscale) === 'undefined') { convert_grayscale = false; }

    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext("2d");

    // due to a Firefox bug
    try {
        ctx.drawImage(img, 0, 0);
    } catch (e) {
        if (e.name === "NS_ERROR_NOT_AVAILABLE") {
            // sometimes happens, lets just abort
            return false;
        } else {
            throw e;
        }
    }
    let img_data: ImageData;
    try {
        img_data = ctx.getImageData(0, 0, canvas.width, canvas.height);
    } catch (e) {
        if (e.name === 'IndexSizeError') {
            return false; // not sure what causes this sometimes but okay abort
        } else {
            throw e;
        }
    }

    // prepare the input: get pixels and normalize them
    const p = img_data.data;
    const W = img.width;
    const H = img.height;
    const pv = []
    for (let i = 0; i < p.length; i++) {
        pv.push(p[i] / 255.0 - 0.5); // normalize image pixels to [-0.5, 0.5]
    }
    let x = new Vol(W, H, 4, 0.0); //input volume (image)
    x.w = pv;

    if (convert_grayscale) {
        // flatten into depth=1 array
        const x1 = new Vol(W, H, 1, 0.0);
        for (let i = 0; i < W; i++) {
            for (let j = 0; j < H; j++) {
                x1.set(i, j, 0, x.get(i, j, 0));
            }
        }
        x = x1;
    }

    return x;
}
