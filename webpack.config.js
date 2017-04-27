const fs = require("fs");
const path = require("path");
const webpack = require("webpack");
const packageJson = JSON.parse(fs.readFileSync(__dirname + "/package.json").toString());

// for the time versioning process works
// const file = "convnetjs-" + packageJson.version + ".js";
const file = "convnet.js";

module.exports = {
    entry: {
        convnetjs: "./lib/index.js"
    },
    output: {
        path: __dirname + "/build",
        filename: file,
        libraryTarget: "var",
        library: "convnetjs"
    }
};
