const fs = require("fs");
const packageJson = JSON.parse(fs.readFileSync(__dirname + "/package.json").toString());

// for the time versioning process works
// const file = "convnetjs-" + packageJson.version + ".js";
const file = "convnetjs.js";

module.exports = {
  entry: './lib/index.js',
  output: {
    path: __dirname + "/build",
    filename: file
  }
};
