#include <stdio.h>
#include <vector>
#include <cmath>

#include "ChSystemParallel.h"
#include "ChLcpSystemDescriptorParallel.h"

#include "utils/input_output.h"

using namespace chrono;
using namespace geometry;

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void AddSphereGeometry(ChSharedBodyDEMPtr&   body,
                       double                radius,
                       const ChVector<>&     pos = ChVector<>(0,0,0),
                       const ChQuaternion<>& rot = ChQuaternion<>(1,0,0,0))
{
  body->GetCollisionModel()->AddSphere(radius, pos);

  ChSharedPtr<ChSphereShape> sphere = ChSharedPtr<ChAsset>(new ChSphereShape);
  sphere->GetSphereGeometry().rad = radius;
  sphere->Pos = pos;
  sphere->Rot = rot;

  body->GetAssets().push_back(sphere);
}

void AddBoxGeometry(ChSharedBodyDEMPtr&    body,
                    const ChVector<>&      hdim,
                    const ChVector<>&      pos = ChVector<>(0,0,0),
                    const ChQuaternion<>&  rot = ChQuaternion<>(1,0,0,0))
{
  // Append to collision geometry
  body->GetCollisionModel()->AddBox(hdim.x, hdim.y, hdim.z, pos, rot);

  // Append to assets
  ChSharedPtr<ChBoxShape> box_shape = ChSharedPtr<ChAsset>(new ChBoxShape);
  box_shape->GetBoxGeometry().Size = hdim;
  box_shape->Pos = pos;
  box_shape->Rot = rot;

  body->GetAssets().push_back(box_shape);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void AddContainer(ChSystemParallelDEM* sys)
{
  // IDs for the two bodies
  int  binId = -200;
  int  mixerId = -201;

  // Create a common material
  ChSharedPtr<ChMaterialSurfaceDEM> mat(new ChMaterialSurfaceDEM);
  mat->SetYoungModulus(2e5f);
  mat->SetFriction(0.4f);
  mat->SetDissipationFactor(0.6f);

  // Create the containing bin (2 x 2 x 1)
  ChSharedBodyDEMPtr bin(new ChBodyDEM(new ChCollisionModelParallel));
  bin->SetMaterialSurfaceDEM(mat);
  bin->SetIdentifier(binId);
  bin->SetMass(1);
  bin->SetPos(ChVector<>(0,0,0));
  bin->SetRot(ChQuaternion<>(1,0,0,0));
  bin->SetCollide(true);
  bin->SetBodyFixed(true);

  ChVector<> hdim(1, 1, 0.5);
  double     hthick = 0.1;

  bin->GetCollisionModel()->ClearModel();
  AddBoxGeometry(bin, ChVector<>(hdim.x, hdim.y, hthick), ChVector<>(0, 0, -hthick));
  AddBoxGeometry(bin, ChVector<>(hthick, hdim.y, hdim.z), ChVector<>(-hdim.x-hthick, 0, hdim.z));
  AddBoxGeometry(bin, ChVector<>(hthick, hdim.y, hdim.z), ChVector<>( hdim.x+hthick, 0, hdim.z));
  AddBoxGeometry(bin, ChVector<>(hdim.x, hthick, hdim.z), ChVector<>(0, -hdim.y-hthick, hdim.z));
  AddBoxGeometry(bin, ChVector<>(hdim.x, hthick, hdim.z), ChVector<>(0,  hdim.y+hthick, hdim.z));
  bin->GetCollisionModel()->BuildModel();

  sys->AddBody(bin);

  // The rotating mixer body (1.6 x 0.2 x 0.4)
  ChSharedBodyDEMPtr mixer(new ChBodyDEM(new ChCollisionModelParallel));
  mixer->SetMaterialSurfaceDEM(mat);
  mixer->SetIdentifier(mixerId);
  mixer->SetMass(10.0);
  mixer->SetInertiaXX(ChVector<>(50,50,50));
  mixer->SetPos(ChVector<>(0,0,0.205));
  mixer->SetBodyFixed(false);
  mixer->SetCollide(true);

  ChVector<> hsize(0.8, 0.1, 0.2);

  mixer->GetCollisionModel()->ClearModel();
  AddBoxGeometry(mixer, hsize);
  mixer->GetCollisionModel()->BuildModel();

  sys->AddBody(mixer);

  // Create an engine between the two bodies
  ChSharedPtr<ChLinkEngine> motor(new ChLinkEngine);

  ChSharedPtr<ChBody> mixer_(mixer);
  ChSharedPtr<ChBody> bin_(bin);

  motor->Initialize(mixer_, bin_,
                    ChCoordsys<>(ChVector<>(0,0,0), ChQuaternion<>(1,0,0,0)));
  motor->Set_eng_mode(ChLinkEngine::ENG_MODE_SPEED);
  if (ChFunction_Const* mfun = dynamic_cast<ChFunction_Const*>(motor->Get_spe_funct()))
    mfun->Set_yconst(CH_C_PI/2); // speed w=90�/s

  sys->AddLink(motor);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void AddFallingBalls(ChSystemParallelDEM* sys)
{
  // Common material
  ChSharedPtr<ChMaterialSurfaceDEM> ballMat(new ChMaterialSurfaceDEM);
  ballMat->SetYoungModulus(2e5f);
  ballMat->SetFriction(0.4f);
  ballMat->SetDissipationFactor(0.6f);

  // Create the falling balls
  int        ballId = 0;
  double     mass = 1;
  double     radius = 0.15;
  ChVector<> inertia = (2.0/5.0)*mass*radius*radius*ChVector<>(1,1,1);

  for (int ix = -2; ix < 3; ix++) {
    for (int iy = -2; iy < 3; iy++) {
      ChVector<> pos(0.4 * ix, 0.4 * iy, 1);

      ChSharedBodyDEMPtr ball(new ChBodyDEM(new ChCollisionModelParallel));
      ball->SetMaterialSurfaceDEM(ballMat);

      ball->SetIdentifier(ballId++);
      ball->SetMass(mass);
      ball->SetInertiaXX(inertia);
      ball->SetPos(pos);
      ball->SetRot(ChQuaternion<>(1, 0, 0, 0));
      ball->SetBodyFixed(false);
      ball->SetCollide(true);

      ball->GetCollisionModel()->ClearModel();
      AddSphereGeometry(ball, radius);
      ball->GetCollisionModel()->BuildModel();

      sys->AddBody(ball);
    }
  }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  int threads = 8;

  // Simulation parameters
  // ---------------------

  double gravity = 9.81;
  double time_step = 1e-4;
  double time_end = 1;
  int    num_steps = std::ceil(time_end / time_step);

  uint max_iteration = 50;
  real tolerance = 1e-8;

  const char* out_folder = "../MIXER_DEM/POVRAY";
  double out_fps = 50;
  int out_steps = std::ceil((1 / time_step) / out_fps);


  // Create system
  // -------------

  ChSystemParallelDEM msystem;

  // Set number of threads.
  int max_threads = msystem.GetParallelThreadNumber();
  if (threads > max_threads)
    threads = max_threads;
  msystem.SetParallelThreadNumber(threads);
  omp_set_num_threads(threads);

  // Set gravitational acceleration
  msystem.Set_G_acc(ChVector<>(0, 0, -gravity));

  // Set solver parameters
  msystem.SetMaxiter(max_iteration);
  msystem.SetIterLCPmaxItersSpeed(max_iteration);
  msystem.SetTol(1e-3);
  msystem.SetTolSpeeds(1e-3);
  msystem.SetStep(time_step);

  ((ChCollisionSystemParallel*) msystem.GetCollisionSystem())->setBinsPerAxis(I3(10, 10, 10));
  ((ChCollisionSystemParallel*) msystem.GetCollisionSystem())->setBodyPerBin(100, 50);

  ((ChCollisionSystemParallel*) msystem.GetCollisionSystem())->ChangeNarrowphase(new ChCNarrowphaseR);

  // Set tolerance and maximum number of iterations for bilateral constraint solver
  ((ChLcpSolverParallelDEM*) msystem.GetLcpSolverSpeed())->SetMaxIteration(max_iteration);
  ((ChLcpSolverParallelDEM*) msystem.GetLcpSolverSpeed())->SetTolerance(tolerance);


  // Create the fixed and moving bodies
  // ----------------------------------

  AddContainer(&msystem);
  AddFallingBalls(&msystem);

  // Perform the simulation
  // ----------------------

  double time = 0;
  int out_frame = 0;
  char filename[100];

  for (int i = 0; i < num_steps; i++) {

    if (i % out_steps == 0) {
      sprintf(filename, "%s/data_%03d.dat", out_folder, out_frame);
      utils::WriteShapesPovray(&msystem, filename);
      out_frame++;
    }

    msystem.DoStepDynamics(time_step);
    time += time_step;
  }

  return 0;
}

