
var loadFile = function(event) {
var reader = new FileReader();
reader.onload = function(){
  var preview = document.getElementById('preview_img');
  centerCrop(reader.result)
    .then(resize)
    .then(function (src) {
      preview.src = src;
    });
};
reader.readAsDataURL(event.target.files[0]);
};

function centerCrop(src){
  return new Promise(function (resolve) {
    var image = new Image();
    image.src = src;
    image.onload = function () {
      var max_width = Math.min(image.width, image.height);
      var max_height = Math.min(image.width, image.height);

      var canvas = document.createElement('canvas');
      var ctx = canvas.getContext("2d");

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      canvas.width = max_width;
      canvas.height = max_height;
      ctx.drawImage(image, (max_width - image.width)/2, (max_height - image.height)/2, image.width, image.height);
      resolve(canvas.toDataURL("image/png"));
    };
  });
}

function resize(src){
  var image = new Image();
  image.src = src;
  return new Promise(function (resolve) {
    image.onload = function () {
      var canvas = document.createElement('canvas');
      canvas.width = image.width;
      canvas.height = image.height;
      var ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0, image.width, image.height);

      var dst = document.createElement('canvas');
      dst.width = image_dimension;
      dst.height = image_dimension;

      window.pica.WW = false;
      window.pica.resizeCanvas(canvas, dst, {
      quality: 2,
      unsharpAmount: 500,
      unsharpThreshold: 100,
      transferable: false
    }, function (err) {  });
      window.pica.WW = true;
      resolve(dst.toDataURL("image/png"));
    };
  });
}
