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
// Description: This file contains all of the setting structures that are used
// within chrono parallel. The constructor for each struct should specify
// default values for every setting parameter that is used by default
// =============================================================================

#ifndef CH_SETTINGS_H
#define CH_SETTINGS_H

#include "chrono_parallel/ChParallelDefines.h"
#include "parallel/ChOpenMP.h"

namespace chrono {
// collision_settings, like the name implies is the structure that contains all
// settings associated with the collision detection step of chrono parallel
struct collision_settings {
  // The default values are specified in the constructor, use these as
  // guidelines when experimenting with your own simulation setup.
  collision_settings() {
    // by default the bounding box is not active and the default values for the
    // bounding box size are not specified.
    use_aabb_active = false;
    // I chose to set the default envelope because there is no good default
    // value (in my opinion, feel free to disagree!) I suggest that the envelope
    // be set to 5-10 percent of the smallest object radius. Using too large of
    // a value will slow the collision detection down.
    collision_envelope = 0;
    // The number of slices in each direction will greatly effect the speed at
    // which the collision detection operates. I would suggest that on average
    // the number of objects in a bin/grid cell should not exceed 100.
    // NOTE!!! this really depends on the architecture that you run on and how
    // many cores you are using.
    bins_per_axis = I3(20, 20, 20);
    narrowphase_algorithm = NARROWPHASE_HYBRID_MPR;
    grid_density = 5;
    leaf_density = 1;
    fixed_bins = true;
    use_two_level = true;
  }

  real3 min_bounding_point, max_bounding_point;
  // This parameter, similar to the one in chrono inflates each collision shape
  // by a certain amount. This is necessary when using DVI as it creates the
  // contact constraints before objects acutally come into contact. In general
  // this helps with stability.
  real collision_envelope;
  // Chrono parallel has an optional feature that allows the user to set a
  // bounding box that automatically freezes (makes inactive) any object that
  // exits the bounding box
  bool use_aabb_active;
  // The size of the bounding box (if set to active) is specified by its min and
  // max extents
  real3 aabb_min, aabb_max;
  // This variable is the primary method to control the granularity of the
  // collision detection grid used for the broadphase.
  // As the name suggests, it is the number of slices along each axis. During
  // the broadphase stage the extents of the simulation are computed and then
  // sliced according to the variable.
  int3 bins_per_axis;
  // There are multiple narrowphase algorithms implemented in the collision
  // detection code. The narrowphase_algorithm parameter can be used to change
  // the type of narrowphase used at runtime.
  NARROWPHASETYPE narrowphase_algorithm;
  real grid_density;
  real leaf_density;
  // use fixed number of bins instead of tuning them
  bool fixed_bins;
  bool use_two_level;
};

// Fluid settings structure
// Currently only one phase is supported
struct fluid_settings {
  fluid_settings() {
    kernel_radius = .04;
    volume = 4.0 / 3.0 * 3.14159265359 * pow(kernel_radius, 3);
    compliance = 0;
    epsilon = 10e-3;
    tau = 4 * .01;
    tau_density = 4 * .01;
    cohesion = 0;
    mu = 0;
    density = 1000;
    mass = 0.037037;
    fluid_is_rigid = true;
    max_velocity = 3;
    viscosity = 0;
    max_interactions = 15;
    collision_envelope = 0;
    contact_recovery_speed = 1;
    artificial_pressure = false;
    artificial_pressure_k = 0.1;
    artificial_pressure_dq = .2 * kernel_radius;
    artificial_pressure_n = 4;
    enable_viscosity = false;
  }
  real kernel_radius;
  real volume;
  real compliance;
  real epsilon;  // Regularization parameter
  real tau;      // Constraint relaxation time
  real tau_density;
  real cohesion;
  real mu;  // friction
  real density;
  real mass;
  real viscosity;
  real collision_envelope;
  bool fluid_is_rigid;
  real max_velocity;            // limit on the maximum speed the fluid can move at
  int max_interactions;         // maximum neighbors supported, increase as needed
  real contact_recovery_speed;  // The speed at which 'rigid' fluid  bodies resolve contact
  bool artificial_pressure;     // Enable artificial pressure term
  real artificial_pressure_k;
  real artificial_pressure_n;
  real artificial_pressure_dq;
  bool enable_viscosity;
};

// solver_settings, like the name implies is the structure that contains all
// settings associated with the parallel solver.
struct solver_settings {
  solver_settings() {
    tolerance = 1e-4;
    tol_speed = 1e-4;
    tolerance_objective = 1e-6;
    collision_in_solver = false;
    update_rhs = false;
    verbose = false;
    test_objective = false;

    alpha = .2;
    omega = .5;
    contact_recovery_speed = .6;
    bilateral_clamp_speed = .6;
    cohesion_epsilon = 0;
    clamp_bilaterals = true;
    perform_stabilization = false;
    collision_in_solver = false;
    presolve = false;
    compute_N = false;
    use_full_inertia_tensor = true;
    max_iteration = 100;
    max_iteration_normal = 0;
    max_iteration_sliding = 100;
    max_iteration_spinning = 0;
    max_iteration_bilateral = 100;

    solver_type = APGD;
    solver_mode = SLIDING;
    local_solver_mode = NORMAL;

    contact_force_model = HERTZ;
    tangential_displ_mode = ONE_STEP;
    use_material_properties = true;
    characteristic_vel = 1;
    min_slip_vel = 1e-4;
    cache_step_length = false;
    skip_tolerance_check = 4;
    apgd_adaptive_step = true;
    apgd_step_size = 1.0;
  }

  // The solver type variable defines name of the solver that will be used to
  // solve the DVI problem
  SOLVERTYPE solver_type;
  // There are three possible solver modes
  // NORMAL will only solve for the normal and bilateral constraints.
  // SLIDING will only solve for the normal, sliding and bilateral constraints.
  // SPINNING will solve for all of the constraints.
  // The purpose of this settings is to allow the user to completely ignore
  // different types of friction. In chrono parallel all constraints support
  // friction and sliding friction so this is how you can improve performance
  // when you know that you don't need spinning/rolling friction or want to solve
  // a problem frictionless
  SOLVERMODE solver_mode;

  // This should not be set by the user, depending on how the iterations are set
  // The variable is used to specify what type of solve is currently being done
  SOLVERMODE local_solver_mode;

  // this parameter is a constant used when solving a problem with compliance
  real alpha;
  // The contact recovery speed parameter controls how "hard" a contact is
  // enforced when two objects are penetrating. The larger the value is the
  // faster the two objects will separate
  real contact_recovery_speed;
  // This parameter is the same as the one for contacts, it controls how fast two
  // objects will move in order to resolve constraint drift.
  real bilateral_clamp_speed;
  // It is possible to disable clamping for bilaterals entirely. When set to true
  // bilateral_clamp_speed is ignored
  bool clamp_bilaterals;
  // An extra prestabilization can be performed for bilaterals before the normal
  // solve, in some cases this will improve the stability of bilateral
  // bilateral constraints
  bool perform_stabilization;
  // Experimental options that probably don't work for all solvers
  bool collision_in_solver;
  bool update_rhs;
  bool presolve;
  bool compute_N;
  bool verbose;
  bool test_objective;
  real cohesion_epsilon;
  bool cache_step_length;
  bool use_full_inertia_tensor;

  // Contact force model for DEM
  CONTACTFORCEMODEL contact_force_model;
  // Tangential contact displacement history. NONE indicates no tangential stiffness,
  // ONE_STEP indicates estimating tangential displacement using only current velocity,
  // MULTI_STEP uses full contact history over multiple steps.
  TANGENTIALDISPLACEMENTMODE tangential_displ_mode;
  // Flag specifying how the stiffness and damping coefficients in the DEM contact
  // force models are calculated. If true, these coefficients are derived from
  // physical material properties. Otherwise, the user specifies the coefficients
  // directly.
  bool use_material_properties;
  // Characteristic velocity (Hooke contact force model)
  real characteristic_vel;
  // Threshold tangential velocity
  real min_slip_vel;

  // Along with setting the solver mode, the total number of iterations for each
  // type of constraints can be performed.
  uint max_iteration;
  // If the normal iterations are set, iterations are performed for just the
  // normal part of the constraints. This will essentially precondition the
  // solver and stiffen the contacts, making objects penetrate less. For visual
  // accuracy this is really useful in my opinion. Bilaterals are still solved
  uint max_iteration_normal;
  // Similarly sliding iterations are only performed on the sliding constraints.
  // Bilaterals are still solved
  uint max_iteration_sliding;
  // Similarly spinning iterations are only performed on the spinning constraints
  // Bilaterals are still solved
  uint max_iteration_spinning;
  uint max_iteration_bilateral;

  // This variable is the tolerance for the solver in terms of speeds
  real tolerance;
  real tol_speed;
  // This variable defines the tolerance if the solver is using the objective
  // termination condition
  real tolerance_objective;
  int skip_tolerance_check;
  //for jacobi and pgs solvers
  real omega;
  bool apgd_adaptive_step;
  real apgd_step_size;
};

struct settings_container {
  settings_container() {
    // The default minimum number of threads is 1, set this to your max threads
    // if you already know that it will run the fastest with that configuration.
    // In some cases simulations start with a few objects but more are added
    // later. By setting it to a low value initially the simulation will run
    // as more objects are added chrono parallel will automatically increase
    // the number of threads. If min threads is > max threads, weird stuff might
    // happen!
    min_threads = 1;
    // The default maximum threads is equal to the number visible via openMP
    max_threads = CHOMPfunctions::GetNumProcs();
    // Only perform thread tuning if max threads is greater than min_threads;
    // I don't really check to see if max_threads is > than min_threads
    // not sure if that is a huge issue
    perform_thread_tuning = ((min_threads == max_threads) ? false : true);
    system_type = SYSTEM_DVI;
    step_size = .01;
  }

  // The settings for the collision detection
  collision_settings collision;
  // The settings for the solver
  solver_settings solver;
  // The settings for the fluid
  fluid_settings fluid;

  // System level settings
  // If set to true chrono parallel will automatically check to see if increasing
  // the number of threads will improve performance. If performance is improved
  // it changes the number of threads, if not, it decreases the number of threads
  // back to the original value.
  bool perform_thread_tuning;
  // The minimum number of threads that will ever be used by this simulation.
  // If you know a good number of threads for your simulation set the minimum so
  // that the simulation is running optimally from the start
  int min_threads;
  // This is the number of threads that the simulation will not exceed
  int max_threads;
  // The timestep of the simulation. This value is copied from chrono currently,
  // setting it has no effect.
  real step_size;

  // The system type defines if the system is solving the DVI frictional contact
  // problem or a DEM penalty based
  SYSTEMTYPE system_type;
};
}

#endif
