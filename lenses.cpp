#include <cstdlib>
//#include <iostream>

#include "lenses.h"

// Implement lens equation, given the lens position (xl, yl) and the
// lens system configuration, shoot a ray back to the source position
// (xs, ys)
void shoot(float& xs, float& ys, float xl, float yl, 
	   float* xlens, float* ylens, float* eps, int nlenses)
{
  float dx, dy, dr;
  xs = xl;
  ys = yl;
  for (int p = 0; p < nlenses; ++p) {
    dx = xl - xlens[p];
    dy = yl - ylens[p];
    dr = dx * dx + dy * dy;
    xs -= eps[p] * dx / dr;
    ys -= eps[p] * dy / dr;
  }

}

// Set up a single lens example
int set_example_1(float** xlens, float** ylens, float** eps)
{
  const int nlenses = 1;
  *xlens = new float[nlenses];
  *ylens = new float[nlenses];
  *eps   = new float[nlenses];

  *xlens[0] = 0.0;
  *ylens[0] = 0.0;
  *eps[0] = 1.0;

  return nlenses;
}

// Simple binary lens
int set_example_2(float** xlens, float** ylens, float** eps)
{
  const int nlenses = 2;
  float* x = new float[nlenses];
  float* y = new float[nlenses];
  float* e = new float[nlenses];
  
  const float eps1 = 0.2;
  x[0] = -0.4; y[0] = 0.0; e[0] = 1 - eps1;
  x[1] =  0.6; y[1] = 0.2; e[1] = eps1;

  *xlens = x;
  *ylens = y;
  *eps = e;
  return nlenses;
}

// Triple lens
int set_example_3(float** xlens, float** ylens, float** eps)
{
  const int nlenses = 3;
  float* x = new float[nlenses];
  float* y = new float[nlenses];
  float* e = new float[nlenses];
  
  const float eps1 = 0.3;
  const float eps2 = 0.2;

  x[0] = -0.4; y[0] = 0.0; e[0] = 1 - eps1 - eps2;
  x[1] =  0.5; y[1] = 0.0; e[1] = eps1;
  x[2] =  0.0; y[2] = 0.4; e[2] = eps2;

  *xlens = x;
  *ylens = y;
  *eps = e;
  return nlenses;
}

float pick_random(float x1, float x2)
{
  float f = rand() / static_cast<float>(RAND_MAX);
  return x1 + f * (x2 - x1);
}

// Many lenses
int set_example_n(const int nuse, float** xlens, float** ylens, float** eps)
{
  const int nlenses = nuse;
  float* x = new float[nlenses];
  float* y = new float[nlenses];
  float* e = new float[nlenses];
  
  float sume = 0;
  const float w = 1.2;
  for (int n =0; n < nlenses; ++n) {
    x[n] = pick_random(-w, w);
    y[n] = pick_random(-w, w);
    e[n] = pick_random(0, 1.0);
    sume += e[n];
  }
  
  // Normalize the mass fractions
  for (int n =0; n < nlenses; ++n) e[n] /= sume;  

  *xlens = x;
  *ylens = y;
  *eps = e;

  return nlenses;
}

