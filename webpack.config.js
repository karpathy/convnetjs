const fs = require("fs");
const packageJson = JSON.parse(fs.readFileSync(__dirname + "/package.json").toString());

const file = "neuralnetworkjs-" + packageJson.version + ".js";

module.exports = {
  entry: './lib/index.js',
  output: {
    path: __dirname + "/build",
    filename: file
  }
};
