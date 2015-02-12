// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All right reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Hammad Mazhar
// =============================================================================
//
// ChronoParallel unit test for MPR collision detection
// =============================================================================
//not used but prevents compilation errors with cuda 7 RC
#include <thrust/transform.h>

#include <stdio.h>
#include <vector>
#include <cmath>
#include <omp.h>

#include "chrono_parallel/constraints/ChConstraintRigidRigid.h"
#include "collision/ChCCollisionModel.h"
#include "core/ChMathematics.h"

#include "unit_testing.h"
//#include <blaze/math/SymmetricMatrix.h>

using namespace std;
using namespace chrono;
using namespace chrono::collision;

double timestep = .001;
double factor = 1.0 / timestep;

int main(int argc,
         char* argv[]) {

   return 0;

}

