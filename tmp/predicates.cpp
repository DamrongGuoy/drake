/*************************************************************************
 *                                                                       *
 * Vega FEM Simulation Library Version 4.0                               *
 *                                                                       *
 * "mesh" library , Copyright (C) 2018 USC                               *
 * All rights reserved.                                                  *
 *                                                                       *
 * Code authors: Yijing Li, Jernej Barbic                                *
 * http://www.jernejbarbic.com/vega                                      *
 *                                                                       *
 * Research: Jernej Barbic, Hongyi Xu, Yijing Li,                        *
 *           Danyong Zhao, Bohan Wang,                                   *
 *           Fun Shing Sin, Daniel Schroeder,                            *
 *           Doug L. James, Jovan Popovic                                *
 *                                                                       *
 * Funding: National Science Foundation, Link Foundation,                *
 *          Singapore-MIT GAMBIT Game Lab,                               *
 *          Zumberge Research and Innovation Fund at USC,                *
 *          Sloan Foundation, Okawa Foundation,                          *
 *          USC Annenberg Foundation                                     *
 *                                                                       *
 * This library is free software; you can redistribute it and/or         *
 * modify it under the terms of the BSD-style license that is            *
 * included with this library in the file LICENSE.txt                    *
 *                                                                       *
 * This library is distributed in the hope that it will be useful,       *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the file     *
 * LICENSE.TXT for more details.                                         *
 *                                                                       *
 *************************************************************************/

#include "predicates.h"
#include <cassert>
#include <functional>
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace vegafem
{
using namespace std;

// use predicates from Shewchuk's predicates
extern "C" void exactinit();
extern "C" double orient2d(double* pa, double *pb, double *pc);
extern "C" double orient3d(double* pa, double *pb, double *pc, double *pd);
extern "C" double incircle(double* pa, double *pb, double *pc, double *pd);
extern "C" double insphere(double* pa, double *pb, double *pc, double *pd, double *pe);

extern "C" double orient2dfast(double* pa, double *pb, double *pc);
extern "C" double orient3dfast(double* pa, double *pb, double *pc, double *pd);
extern "C" double incirclefast(double* pa, double *pb, double *pc, double *pd);
extern "C" double inspherefast(double* pa, double *pb, double *pc, double *pd, double *pe);


// -------------------------------------------------------------

void initPredicates()
{
  exactinit();
}

double orient2d(const double pa[2], const double pb[2], const double pc[2])
{
  return orient2d((double*)pa, (double*)pb, (double*)pc);
}

double orient3d(const double pa[3], const double pb[3], const double pc[3], const double pd[3])
{
  return orient3d((double*)pa, (double*)pb, (double*)pc, (double*)pd);
}

double incircle(const double pa[2], const double pb[2], const double pc[2], const double pd[2])
{
  return incircle((double*)pa, (double*)pb, (double*)pc, (double*)pd);
}

double insphere(const double pa[3], const double pb[3], const double pc[3], const double pd[3], const double pe[3])
{
  return insphere((double*)pa, (double*)pb, (double*)pc, (double*)pd, (double*)pe);
}

double orient2dfast(const double pa[2], const double pb[2], const double pc[2])
{
  return orient2dfast((double*)pa, (double*)pb, (double*)pc);
}

double orient3dfast(const double pa[3], const double pb[3], const double pc[3], const double pd[3])
{
  return orient3dfast((double*)pa, (double*)pb, (double*)pc, (double*)pd);
}

double incirclefast(const double pa[2], const double pb[2], const double pc[2], const double pd[2])
{
  return incirclefast((double*)pa, (double*)pb, (double*)pc, (double*)pd);
}

double inspherefast(const double pa[3], const double pb[3], const double pc[3], const double pd[3], const double pe[3])
{
  return inspherefast((double*)pa, (double*)pb, (double*)pc, (double*)pd, (double*)pe);
}

bool intersectTriTet(const double tria[3], const double trib[3], const double tric[3],
    const double teta[3], const double tetb[3], const double tetc[3], const double tetd[3])
{
  return pointInTet(tria, teta, tetb, tetc, tetd) ||
         // triangle edges against tet faces
         intersectSegTri(tria, trib, teta, tetb, tetc) || intersectSegTri(tria, trib, teta, tetb, tetd) ||
         intersectSegTri(tria, trib, teta, tetc, tetd) || intersectSegTri(tria, trib, tetb, tetc, tetd) ||
         intersectSegTri(tria, tric, teta, tetb, tetc) || intersectSegTri(tria, tric, teta, tetb, tetd) ||
         intersectSegTri(tria, tric, teta, tetc, tetd) || intersectSegTri(tria, tric, tetb, tetc, tetd) ||
         intersectSegTri(trib, tric, teta, tetb, tetc) || intersectSegTri(trib, tric, teta, tetb, tetd) ||
         intersectSegTri(trib, tric, teta, tetc, tetd) || intersectSegTri(trib, tric, tetb, tetc, tetd) ||
         // tet edges against triangle
         intersectSegTri(teta, tetb, tria, trib, tric) || intersectSegTri(teta, tetc, tria, trib, tric) ||
         intersectSegTri(teta, tetd, tria, trib, tric) || intersectSegTri(tetb, tetc, tria, trib, tric) ||
         intersectSegTri(tetb, tetd, tria, trib, tric) || intersectSegTri(tetc, tetd, tria, trib, tric);
}




namespace
{

void getXY(const double v[3], double o[2]) { o[0] = v[0]; o[1] = v[1]; }
void getYZ(const double v[3], double o[2]) { o[0] = v[1]; o[1] = v[2]; }
void getZX(const double v[3], double o[2]) { o[0] = v[2]; o[1] = v[0]; }
std::function<void(const double v[3], double o[2])> getComp[3] = { getXY, getYZ, getZX };

}

bool isTriangleDegenerate(const double ta[3], const double tb[3], const double tc[3])
{
  for(int i = 0; i < 3; i++)
  {
    auto f = getComp[i];
    double a[2], b[2], c[2];
    f(ta, a);
    f(tb, b);
    f(tc, c);

    if (orient2d(a,b,c) != 0.0) return false;
  }
  return true;
}

int inCircumsphereOnPlane(const double ta[3], const double tb[3], const double tc[3], const double td[3])
{
  for(int i = 0; i < 3; i++)
  {
    auto f = getComp[i];
    double a[2], b[2], c[2], d[2];
    f(ta, a);
    f(tb, b);
    f(tc, c);
    f(td, d);
    double ret = incircle(a, b, c, d);
    if (ret < 0) return -1;
    if (ret > 0) return 1;
  }
  return 0;
}


}//namespace vegafem
