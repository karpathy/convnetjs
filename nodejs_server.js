var connect = require('connect');

// just serve local files
connect.createServer(
    connect.static(__dirname)
).listen(8080);

