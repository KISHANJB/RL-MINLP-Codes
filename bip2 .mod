# bip2.mod
var x1 binary;
var x2 binary;

maximize obj: 3*x1 + 5*x2;

subject to constraint1:
    2*x1 + 3*x2 <= 4;

