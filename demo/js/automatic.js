 
    // utility functions
    Array.prototype.contains = function(v) {
    for(var i = 0; i < this.length; i++) {
      if(this[i] === v) return true;
    }
    return false;
    };
    Array.prototype.unique = function() {
      var arr = [];
      for(var i = 0; i < this.length; i++) {
        if(!arr.contains(this[i])) {
          arr.push(this[i]);
        }
      }
      return arr; 
    }
    
    function FAIL(outdivid, msg) {
      $(outdivid).prepend("<div class=\"msg\" style=\"background-color:#FCC;\">"+msg+"</div>")
    }
    function SUCC(outdivid, msg) {
      $(outdivid).prepend("<div class=\"msg\" style=\"background-color:#CFC;\">"+msg+"</div>")
    }

    // looks at a column i of data and guesses what's in it
    // returns results of analysis: is column numeric? How many unique entries and what are they?
    function guessColumn(data, c) {
      var numeric = true;
      var vs = [];
      for(var i=0,n=data.length;i<n;i++) {
        var v = data[i][c];
        vs.push(v);
        if(isNaN(v)) numeric = false;
      }
      var u = vs.unique();
      if(!numeric) {
        // if we have a non-numeric we will map it through uniques to an index
        return {numeric:numeric, num:u.length, uniques:u};
      } else {
        return {numeric:numeric, num:u.length};
      }
    }
    
    // returns arr (csv parse)
    // and colstats, which contains statistics about the columns of the input
    // parsing results will be appended to a div with id outdivid
    function importData(arr, outdivid) {
      $(outdivid).empty(); // flush messages

      // find number of datapoints
      N = arr.length;
      var t = [];
      SUCC(outdivid, "found " + N + " data points");
      if(N === 0) { FAIL(outdivid, 'no data points found?'); return; }
      
      // find dimensionality and enforce consistency
      D = arr[0].length;
      for(var i=0;i<N;i++) {
        var d = arr[i].length;
        if(d !== D) { FAIL(outdivid, 'data dimension not constant: line ' + i + ' has ' + d + ' entries.'); return; }
      }
      SUCC(outdivid, "data dimensionality is " + (D-1));
      
      // go through columns of data and figure out what they are
      var colstats = [];
      for(var i=0;i<D;i++) {
        var res = guessColumn(arr, i);
        colstats.push(res);
        if(D > 20 && i>3 && i < D-3) {
          if(i==4) {
            SUCC(outdivid, "..."); // suppress output for too many columns
          }
        } else {
          SUCC(outdivid, "column " + i + " looks " + (res.numeric ? "numeric" : "NOT numeric") + " and has " + res.num + " unique elements");
        }
      }

      return {arr: arr, colstats: colstats};
   }
    
  // process input mess into vols and labels
  function makeDataset(arr, colstats) {

    var labelix = parseInt($("#labelix").val());
    if(labelix < 0) labelix = D + labelix; // -1 should turn to D-1

    var data = [];
    var labels = [];
    for(var i=0;i<N;i++) {
      var arri = arr[i];
      
      // create the input datapoint Vol()
      var p = arri.slice(0, D-1);
      var xarr = [];
      for(var j=0;j<D;j++) {
        if(j===labelix) continue; // skip!

        if(colstats[j].numeric) {
          xarr.push(parseFloat(arri[j]));
        } else {
          var u = colstats[j].uniques;
          var ix = u.indexOf(arri[j]); // turn into 1ofk encoding
          for(var q=0;q<u.length;q++) {
            if(q === ix) { xarr.push(1.0); }
            else { xarr.push(0.0); }
          }
        }
      }
      var x = new convnetjs.Vol(xarr);
      
      // process the label (last column)
      if(colstats[labelix].numeric) {
        var L = parseFloat(arri[labelix]); // regression
      } else {
        var L = colstats[labelix].uniques.indexOf(arri[labelix]); // classification
        if(L==-1) {
          console.log('whoa label not found! CRITICAL ERROR, very fishy.');
        }
      }
      data.push(x);
      labels.push(L);
    }
    
    var dataset = {};
    dataset.data = data;
    dataset.labels = labels;
    return dataset;
  }

  // optionally provide a magic net
  function testEval(optional_net) {
    if (typeof optional_net !== 'undefined') {
      var net = optional_net;
    } else {
      var net = magicNet;
    }

    // set options for magic net
    net.ensemble_size = parseInt($("#ensemblenum").val())

    // read in the data in the text field
    var test_dataset = importTestData();
    // use magic net to predict
    var n = test_dataset.data.length;
    var acc = 0.0;
    for(var i=0;i<n;i++) {
      var yhat = net.predict(test_dataset.data[i]);
      if(yhat === -1) {
        $("#testresult").html("The MagicNet is not yet ready! It must process at least one batch of candidates across all folds first. Wait a bit.");
        $("#testresult").css('background-color', '#FCC');
        return;
      }
      var l = test_dataset.labels[i];
      acc += (yhat === l ? 1 : 0); // 0-1 loss
      console.log('test example ' + i + ': predicting ' + yhat + ', ground truth is ' + l);
    }
    acc /= n;

    // report accuracy
    $("#testresult").html("Test set accuracy: " + acc);
    $("#testresult").css('background-color', '#CFC');
  }

  function reinitGraph() {
    var legend = [];
    for(var i=0;i<magicNet.candidates.length;i++) {
      legend.push('model ' + i);
    }
    valGraph = new cnnvis.MultiGraph(legend, {miny: 0, maxy: 1});
  }

  var folds_evaluated = 0;
  function finishedFold() {
    folds_evaluated++;
    $("#foldreport").html("So far evaluated a total of " + folds_evaluated + "/" + magicNet.num_folds + " folds in current batch");
    reinitGraph();
  }
  var batches_evaluated = 0;
  function finishedBatch() {
    batches_evaluated++;
    $("#candsreport").html("So far evaluated a total of " + batches_evaluated + " batches of candidates");
  }

  var magicNet = null;
  function startCV() { // takes in train_dataset global
    var opts = {}
    opts.train_ratio = parseInt($("#trainp").val())/100.0;
    opts.num_folds = parseInt($("#foldsnum").val());
    opts.num_candidates = parseInt($("#candsnum").val());
    opts.num_epochs = parseInt($("#epochsnum").val());
    opts.neurons_min = parseInt($("#nnmin").val());
    opts.neurons_max = parseInt($("#nnmin").val());
    magicNet = new convnetjs.MagicNet(train_dataset.data, train_dataset.labels, opts);
    magicNet.onFinishFold(finishedFold);
    magicNet.onFinishBatch(finishedBatch);

    folds_evaluated = 0;
    batches_evaluated = 0;
    $("#candsreport").html("So far evaluated a total of " + batches_evaluated + " batches of candidates");
    $("#foldreport").html("So far evaluated a total of " + folds_evaluated + "/" + magicNet.num_folds + " folds in current batch");
    reinitGraph();

    var legend = [];
    for(var i=0;i<magicNet.candidates.length;i++) {
      legend.push('model ' + i);
    }
    valGraph = new cnnvis.MultiGraph(legend, {miny: 0, maxy: 1});
    setInterval(step, 0);
  }
      
    var fold;
    var cands = [];
    var dostep = false;
    var valGraph;
    var iter = 0;
    function step() {
      iter++;
      
      magicNet.step();
      if(iter % 300 == 0) {

        var vals = magicNet.evalValErrors();
        valGraph.add(magicNet.iter, vals);
        valGraph.drawSelf(document.getElementById("valgraph"));
    
        // print out the best models so far
        var cands = magicNet.candidates; // naughty: get pointer to internal data
        var scores = [];
        for(var k=0;k<cands.length;k++) {
          var c = cands[k];
          var s = c.acc.length === 0 ? 0 : c.accv / c.acc.length;
          scores.push(s);
        }
        var mm = convnetjs.maxmin(scores);
        var cm = cands[mm.maxi];
        var t = '';
        if(c.acc.length > 0) {
          t += 'Results based on ' + c.acc.length + ' folds:';
          t += 'best model in current batch (validation accuracy ' + mm.maxv + '):<br>';
          t += '<b>Net layer definitions:</b><br>';
          t += JSON.stringify(cm.layer_defs);
          t += '<br><b>Trainer definition:</b><br>';
          t += JSON.stringify(cm.trainer_def);
          t += '<br>';
        }
        $('#bestmodel').html(t);

        // also print out the best model so far
        var t = '';
        if(magicNet.evaluated_candidates.length > 0) {
          var cm = magicNet.evaluated_candidates[0];
          t += 'validation accuracy of best model so far, overall: ' + cm.accv / cm.acc.length + '<br>';
          t += '<b>Net layer definitions:</b><br>';
          t += JSON.stringify(cm.layer_defs);
          t += '<br><b>Trainer definition:</b><br>';
          t += JSON.stringify(cm.trainer_def);
          t += '<br>';
        }
        $('#bestmodeloverall').html(t);
      }
    }
    
    // TODO: MOVE TO CONVNETJS UTILS
    var randperm = function(n) {
      var i = n,
          j = 0,
          temp;
      var array = [];
      for(var q=0;q<n;q++)array[q]=q;
      while (i--) {
          j = Math.floor(Math.random() * (i+1));
          temp = array[i];
          array[i] = array[j];
          array[j] = temp;
      }
      return array;
    }

    var train_dataset, train_import_data; // globals
    function importTrainData() {
      var csv_txt = $('#data-ta').val();
      var arr = $.csv.toArrays(csv_txt);
      var arr_train = arr;
      var arr_test = [];

      var test_ratio = Math.floor($("#testsplit").val());
      if(test_ratio !== 0) {
        // send some lines to test set
        var test_lines_num = Math.floor(arr.length * test_ratio / 100.0);
        var rp = randperm(arr.length);
        arr_train = [];
        for(var i=0;i<arr.length;i++) {
          if(i<test_lines_num) {
            arr_test.push(arr[rp[i]]);
          } else {
            arr_train.push(arr[rp[i]]);
          }
        }
        // enter test lines to test box
        var t = "";
        for(var i=0;i<arr_test.length;i++) {
          t+= arr_test[i].join(",")+"\n";
        }
        $("#data-te").val(t);
        $("#datamsgtest").empty();
      }

      $("#prepromsg").empty(); // flush
      SUCC("#prepromsg", "Sent " + arr_test.length + " data to test, keeping " + arr_train.length + " for train.");
      train_import_data = importData(arr_train,'#datamsg');
      train_dataset = makeDataset(train_import_data.arr, train_import_data.colstats);
      return train_dataset;
    }

    function importTestData() {
      var csv_txt = $('#data-te').val();
      var arr = $.csv.toArrays(csv_txt);
      var import_data = importData(arr,'#datamsgtest');
      // note important that we use colstats of train data!
      test_dataset = makeDataset(import_data.arr, train_import_data.colstats);
      return test_dataset;
    }

    function loadDB(url) {
      // load a dataset from a url with ajax
      $.ajax({
        url: url,
        dataType: "text",
        success: function(txt) {
          $("#data-ta").val(txt);
        }
      });
    }

    function start() {
      loadDB('data/car.data.txt');
    }

    function exportMagicNet() {
      $("#taexport").val(JSON.stringify(magicNet.toJSON()));

      /*
      // for debugging
      var j = JSON.parse($("#taexport").val());
      var m = new convnetjs.MagicNet();
      m.fromJSON(j);
      testEval(m);
      */
    }

    function changeNNRange() {
      magicNet.neurons_min = parseInt($("#nnmin").val());
      magicNet.neurons_max = parseInt($("#nnmax").val());
    }
