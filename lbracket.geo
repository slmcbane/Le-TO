sz = 0.02;

//+
Point(1) = {0, 0, 0, sz};
//+
Point(2) = {0.4, 1, 0, sz};
//+
Point(3) = {0.4, 0.4, 0, sz};
//+
Point(4) = {0.9, 0.4, 0, sz};
//+
Point(5) = {1.0, 0.4, 0, sz};
//+
Point(6) = {1.0, 0.0, 0, sz};
//+
Point(7) = {0.0, 1.0, 0, sz};
//+
Line(1) = {1, 7};
//+
Line(2) = {7, 2};
//+
Line(3) = {2, 3};
//+
Line(4) = {3, 4};
//+
Line(5) = {4, 5};
//+
Line(6) = {5, 6};
//+
Line(7) = {6, 1};
//+
Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7};
//+
Plane Surface(1) = {1};
//+
Physical Curve("ESSENTIAL") = {2};
//+
Physical Curve("FORCE") = {5};
