(function(global) {
  "use strict";

  var convnetjs = global.convnetjs || {};

  function ConfigError(message) {
    this.name = "ConfigError";
    this.message = message;
  }
  ConfigError.prototype = new Error();

  function stripComments(source) {
    var out = "";
    var quote = null;
    for(var i = 0; i < source.length; i++) {
      var c = source.charAt(i);
      var n = source.charAt(i + 1);
      if(quote) {
        out += c;
        if(c === "\\" && i + 1 < source.length) {
          i++;
          out += source.charAt(i);
        } else if(c === quote) {
          quote = null;
        }
      } else if(c === "'" || c === '"') {
        quote = c;
        out += c;
      } else if(c === "/" && n === "/") {
        while(i < source.length && source.charAt(i) !== "\n") i++;
        out += "\n";
      } else if(c === "/" && n === "*") {
        i += 2;
        while(i < source.length && !(source.charAt(i) === "*" && source.charAt(i + 1) === "/")) i++;
        i++;
      } else {
        out += c;
      }
    }
    return out;
  }

  function splitStatements(source) {
    var statements = [];
    var start = 0;
    var depth = 0;
    var quote = null;
    for(var i = 0; i < source.length; i++) {
      var c = source.charAt(i);
      if(quote) {
        if(c === "\\" && i + 1 < source.length) i++;
        else if(c === quote) quote = null;
      } else if(c === "'" || c === '"') {
        quote = c;
      } else if(c === "{" || c === "[" || c === "(") {
        depth++;
      } else if(c === "}" || c === "]" || c === ")") {
        depth--;
      } else if(c === ";" && depth === 0) {
        var statement = source.substring(start, i).replace(/^\s+|\s+$/g, "");
        if(statement) statements.push(statement);
        start = i + 1;
      }
    }
    var last = source.substring(start).replace(/^\s+|\s+$/g, "");
    if(last) statements.push(last);
    return statements;
  }

  function Parser(source, scope) {
    this.source = source;
    this.i = 0;
    this.scope = scope;
  }

  Parser.prototype = {
    peek: function() { return this.source.charAt(this.i); },
    skip: function() {
      while(/\s/.test(this.peek())) this.i++;
    },
    expect: function(c) {
      this.skip();
      if(this.peek() !== c) throw new ConfigError("Expected '" + c + "' at column " + this.i);
      this.i++;
    },
    parseIdentifier: function() {
      this.skip();
      var start = this.i;
      if(!/[A-Za-z_$]/.test(this.peek())) throw new ConfigError("Expected identifier at column " + this.i);
      this.i++;
      while(/[A-Za-z0-9_$]/.test(this.peek())) this.i++;
      return this.source.substring(start, this.i);
    },
    parseString: function() {
      this.skip();
      var quote = this.peek();
      if(quote !== "'" && quote !== '"') throw new ConfigError("Expected string at column " + this.i);
      this.i++;
      var out = "";
      while(this.i < this.source.length) {
        var c = this.source.charAt(this.i++);
        if(c === "\\") {
          if(this.i < this.source.length) out += this.source.charAt(this.i++);
        } else if(c === quote) {
          return out;
        } else {
          out += c;
        }
      }
      throw new ConfigError("Unterminated string literal");
    },
    parseNumber: function() {
      this.skip();
      var match = /^[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?/i.exec(this.source.substring(this.i));
      if(!match) throw new ConfigError("Expected number at column " + this.i);
      this.i += match[0].length;
      return parseFloat(match[0]);
    },
    parseArray: function() {
      var arr = [];
      this.expect("[");
      this.skip();
      if(this.peek() === "]") { this.i++; return arr; }
      while(true) {
        arr.push(this.parseExpression());
        this.skip();
        if(this.peek() === "]") { this.i++; return arr; }
        this.expect(",");
      }
    },
    parseObject: function() {
      var obj = {};
      this.expect("{");
      this.skip();
      if(this.peek() === "}") { this.i++; return obj; }
      while(true) {
        this.skip();
        var key = (this.peek() === "'" || this.peek() === '"') ? this.parseString() : this.parseIdentifier();
        this.expect(":");
        obj[key] = this.parseExpression();
        this.skip();
        if(this.peek() === "}") { this.i++; return obj; }
        this.expect(",");
      }
    },
    resolveIdentifier: function(name) {
      if(name === "true") return true;
      if(name === "false") return false;
      if(name === "null") return null;
      if(Object.prototype.hasOwnProperty.call(this.scope, name)) return this.scope[name];
      throw new ConfigError("Unknown identifier: " + name);
    },
    parsePrimary: function() {
      this.skip();
      var c = this.peek();
      if(c === "'" || c === '"') return this.parseString();
      if(c === "[") return this.parseArray();
      if(c === "{") return this.parseObject();
      if(c === "(") {
        this.i++;
        var value = this.parseExpression();
        this.expect(")");
        return value;
      }
      if(/[+\-\d.]/.test(c)) return this.parseNumber();
      return this.resolveIdentifier(this.parseIdentifier());
    },
    parseMul: function() {
      var value = this.parsePrimary();
      while(true) {
        this.skip();
        var op = this.peek();
        if(op !== "*" && op !== "/") return value;
        this.i++;
        var right = this.parsePrimary();
        value = op === "*" ? value * right : value / right;
      }
    },
    parseExpression: function() {
      var value = this.parseMul();
      while(true) {
        this.skip();
        var op = this.peek();
        if(op !== "+" && op !== "-") return value;
        this.i++;
        var right = this.parseMul();
        value = op === "+" ? value + right : value - right;
      }
    }
  };

  function parseExpression(source, scope) {
    var parser = new Parser(source, scope);
    var value = parser.parseExpression();
    parser.skip();
    if(parser.i !== source.length) throw new ConfigError("Unexpected token at column " + parser.i);
    return value;
  }

  function splitArgs(source) {
    return splitStatements(source.replace(/,/g, ";"));
  }

  function execute(source, options) {
    options = options || {};
    var scope = {};
    var statements = splitStatements(stripComments(source));

    for(var i = 0; i < statements.length; i++) {
      var s = statements[i];
      var m;

      m = /^(?:var\s+)?([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*new\s+convnetjs\.Net\(\)$/.exec(s);
      if(m) { scope[m[1]] = new convnetjs.Net(); continue; }

      m = /^(?:var\s+)?([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*new\s+convnetjs\.(?:SGDTrainer|Trainer)\(([^,]+),(.+)\)$/.exec(s);
      if(m) {
        scope[m[1]] = new convnetjs.SGDTrainer(scope[m[2].replace(/\s/g, "")], parseExpression(m[3], scope));
        continue;
      }

      m = /^(?:var\s+)?([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*new\s+deepqlearn\.Brain\((.+)\)$/.exec(s);
      if(m) {
        if(!global.deepqlearn || !global.deepqlearn.Brain) throw new ConfigError("deepqlearn.Brain is not available");
        var args = splitArgs(m[2]);
        scope[m[1]] = new global.deepqlearn.Brain(
          parseExpression(args[0], scope),
          parseExpression(args[1], scope),
          args.length > 2 ? parseExpression(args[2], scope) : undefined
        );
        continue;
      }

      m = /^([A-Za-z_$][A-Za-z0-9_$]*)\.makeLayers\(([^)]+)\)$/.exec(s);
      if(m) {
        if(!scope[m[1]]) throw new ConfigError("Unknown network: " + m[1]);
        scope[m[1]].makeLayers(parseExpression(m[2], scope));
        continue;
      }

      m = /^([A-Za-z_$][A-Za-z0-9_$]*)\.push\((.+)\)$/.exec(s);
      if(m) {
        if(!scope[m[1]] || !scope[m[1]].push) throw new ConfigError("Cannot push into: " + m[1]);
        scope[m[1]].push(parseExpression(m[2], scope));
        continue;
      }

      m = /^([A-Za-z_$][A-Za-z0-9_$]*)\.([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(.+)$/.exec(s);
      if(m) {
        if(!scope[m[1]]) scope[m[1]] = {};
        scope[m[1]][m[2]] = parseExpression(m[3], scope);
        continue;
      }

      m = /^(?:var\s+)?([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(.+)$/.exec(s);
      if(m) {
        scope[m[1]] = parseExpression(m[2], scope);
        continue;
      }

      throw new ConfigError("Unsupported configuration statement: " + s);
    }

    if(options.requireNet && !scope.net) throw new ConfigError("Configuration did not create net");
    if(options.requireTrainer && !scope.trainer) throw new ConfigError("Configuration did not create trainer");
    if(options.requireBrain && !scope.brain) throw new ConfigError("Configuration did not create brain");
    return scope;
  }

  convnetjs.demoConfig = {
    ConfigError: ConfigError,
    execute: execute,
    parseNetwork: function(source) {
      return execute(source, {requireNet: true, requireTrainer: true});
    },
    parseTrainerComparison: function(source) {
      var scope = execute(source);
      if(!scope.layer_defs || !scope.trainer_defs || !scope.legend) {
        throw new ConfigError("Trainer comparison config requires layer_defs, trainer_defs and legend");
      }
      return scope;
    },
    parseDeepQ: function(source) {
      return execute(source, {requireBrain: true});
    }
  };

  global.convnetjs = convnetjs;
})(this);
