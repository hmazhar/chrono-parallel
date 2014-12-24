#include <stdio.h>
#include <vector>
#include <cmath>

#include "core/ChFileutils.h"
#include "core/ChStream.h"

#include "chrono_parallel/physics/ChSystemParallel.h"
#include "chrono_parallel/lcp/ChLcpSystemDescriptorParallel.h"
#include "chrono_parallel/collision/ChCNarrowphaseRUtils.h"

#include "chrono_utils/ChUtilsGeometry.h"
#include "chrono_utils/ChUtilsCreators.h"
#include "chrono_utils/ChUtilsInputOutput.h"

#include "config.h"
#include "subsys/ChVehicleModelData.h"
#include "subsys/vehicle/Vehicle.h"
#include "subsys/powertrain/SimplePowertrain.h"


#ifdef CHRONO_PARALLEL_HAS_OPENGL
#include "chrono_opengl/ChOpenGLWindow.h"
#endif

using namespace chrono;
using namespace chrono::collision;

using std::cout;
using std::endl;


// =============================================================================

// JSON file for vehicle model
std::string vehicle_file("hmmwv/vehicle/HMMWV_Vehicle.json");
//std::string vehicle_file("generic/vehicle/Vehicle_DoubleWishbones.json");
//std::string vehicle_file("generic/vehicle/Vehicle_MultiLinks.json");
//std::string vehicle_file("generic/vehicle/Vehicle_SolidAxles.json");
//std::string vehicle_file("generic/vehicle/Vehicle_ThreeAxles.json");

// JSON files for powertrain (simple)
std::string simplepowertrain_file("hmmwv/powertrain/HMMWV_SimplePowertrain.json");

// Initial vehicle position and orientation
ChVector<> initLoc(0, 0, 1.0);
ChQuaternion<> initRot(1, 0, 0, 0);

// -----------------------------------------------------------------------------
// Simulation parameters
// -----------------------------------------------------------------------------

// Desired number of OpenMP threads (will be clamped to maximum available)
int threads = 100;

// Perform dynamic tuning of number of threads?
bool thread_tuning = true;

// Simulation duration.
double time_end = 5;

// Solver parameters
double time_step = 1e-4;

double tolerance = 1e-3;

int max_iteration_normal = 0;
int max_iteration_sliding = 200;
int max_iteration_spinning = 0;

float contact_recovery_speed = 0.1;

// Output
const std::string out_dir = "../HMMWV";
const std::string pov_dir = out_dir + "/POVRAY";

int out_fps = 60;

// Continuous loop (only if OpenGL available)
bool loop = false;

// =============================================================================
int main(int argc, char* argv[])
{

  // Set path to ChronoVehicle data files
  vehicle::SetDataPath(CHRONOVEHICLE_DATA_DIR);


  // --------------------------
  // Create output directories.
  // --------------------------

  if(ChFileutils::MakeDirectory(out_dir.c_str()) < 0) {
    cout << "Error creating directory " << out_dir << endl;
    return 1;
  }
  if(ChFileutils::MakeDirectory(pov_dir.c_str()) < 0) {
    cout << "Error creating directory " << pov_dir << endl;
    return 1;
  }

  // --------------
  // Create system.
  // --------------

  ChSystemParallelDVI* system = new ChSystemParallelDVI();

  system->Set_G_acc(ChVector<>(0, 0, -9.81));

  // ----------------------
  // Set number of threads.
  // ----------------------

  int max_threads = system->GetParallelThreadNumber();
  if (threads > max_threads)
    threads = max_threads;
  system->SetParallelThreadNumber(threads);
  omp_set_num_threads(threads);
  cout << "Using " << threads << " threads" << endl;

  // ---------------------
  // Edit system settings.
  // ---------------------

  system->GetSettings()->solver.tolerance = tolerance;

  system->GetSettings()->solver.solver_mode = SLIDING;
  system->GetSettings()->solver.max_iteration_normal = max_iteration_normal;
  system->GetSettings()->solver.max_iteration_sliding = max_iteration_sliding;
  system->GetSettings()->solver.max_iteration_spinning = max_iteration_spinning;
  system->GetSettings()->solver.alpha = 0;
  system->GetSettings()->solver.contact_recovery_speed = contact_recovery_speed;
  system->ChangeSolverType(APGDRS);

  system->GetSettings()->collision.bins_per_axis = I3(10, 10, 10);

  // --------------------------------------------------------
  // Create and initialize the vehicle and powertrain systems
  // --------------------------------------------------------

  Vehicle vehicle(system, vehicle::GetDataFile(vehicle_file));
  vehicle.Initialize(ChCoordsys<>(initLoc, initRot));

  // Create and initialize the powertrain system
  SimplePowertrain powertrain(vehicle::GetDataFile(simplepowertrain_file));
  powertrain.Initialize();


  // -----------------------
  // Perform the simulation.
  // -----------------------

#ifdef CHRONO_PARALLEL_HAS_OPENGL
  // Initialize OpenGL
  opengl::ChOpenGLWindow &gl_window = opengl::ChOpenGLWindow::getInstance();
  gl_window.Initialize(1280, 720, "mixerDEM", system);
  gl_window.SetCamera(ChVector<>(0,-10,0), ChVector<>(0,0,0),ChVector<>(0,0,1));

  // Let the OpenGL manager run the simulation until interrupted.
  if (loop) {
    gl_window.StartDrawLoop(time_step);
    return 0;
  }
#endif

  // Run simulation for specified time.
  int out_steps = std::ceil((1.0 / time_step) / out_fps);

  double time = 0;
  int sim_frame = 0;
  int out_frame = 0;
  int next_out_frame = 0;
  double exec_time = 0;
  int num_contacts = 0;

  while (time < time_end) {
    if (sim_frame == next_out_frame) {
      char filename[100];
      sprintf(filename, "%s/data_%03d.dat", pov_dir.c_str(), out_frame + 1);
      utils::WriteShapesPovray(system, filename);

      cout << "------------ Output frame:   " << out_frame << endl;
      cout << "             Sim frame:      " << sim_frame << endl;
      cout << "             Time:           " << time << endl;
      cout << "             Avg. contacts:  " << num_contacts / out_steps << endl;
      cout << "             Execution time: " << exec_time << endl;


      out_frame++;
      next_out_frame += out_steps;
      num_contacts = 0;
    }

    // Advance dynamics.
#ifdef CHRONO_PARALLEL_HAS_OPENGL
    if (gl_window.Active()) {
       gl_window.DoStepDynamics(time_step);
       gl_window.Render();
    }
#else
    system->DoStepDynamics(time_step);
#endif

    // Update counters.
    time += time_step;
    sim_frame++;
    exec_time += system->GetTimerStep();
    num_contacts += system->GetNcontacts();
  }

  // Final stats
  cout << "==================================" << endl;
  cout << "Simulation time:   " << exec_time << endl;
  cout << "Number of threads: " << threads << endl;

  return 0;
}
