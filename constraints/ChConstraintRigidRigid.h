#ifndef CHCONSTRAINT_RIGIDRIGID_H
#define CHCONSTRAINT_RIGIDRIGID_H

#include "ChBaseGPU.h"

namespace chrono {
class ChApiGPU ChConstraintRigidRigid: public ChBaseGPU {
	public:
		ChConstraintRigidRigid(ChGPUDataManager *data_container_) {
			data_container = data_container_;
			Initialize();

			if (number_of_rigid_rigid > 0) {
				update_number.resize((number_of_rigid_rigid) * 2, 0);
				offset_counter.resize((number_of_rigid_rigid) * 2, 0);
				update_offset.resize((number_of_rigid_rigid) * 2, 0);
				body_num.resize((number_of_rigid_rigid) * 2, 0);
				vel_update.resize((number_of_rigid_rigid) * 2);
				omg_update.resize((number_of_rigid_rigid) * 2);

				host_Offsets(data_container->host_data.bids_data.data(), body_num.data());

				Thrust_Sequence(update_number);
				Thrust_Sequence(update_offset);
				Thrust_Fill(offset_counter, 0);
				Thrust_Sort_By_Key(body_num, update_number);
				Thrust_Sort_By_Key(update_number, update_offset);
				body_number = body_num;
				Thrust_Reduce_By_KeyB(number_of_updates, body_num, update_number, offset_counter);
				Thrust_Inclusive_Scan(offset_counter);
			}

		}
		~ChConstraintRigidRigid() {
		}
		void host_Project(int2 *ids, real *friction, real* cohesion, real *gamma);
		void Project(custom_vector<real> & gamma);

		void host_RHS(int2 *ids, real *correction, real * compliance, bool * active, real3 *vel, real3 *omega, real3 *JXYZA, real3 *JXYZB, real3 *JUVWA, real3 *JUVWB, real *rhs);
		void ComputeRHS();

		void host_Jacobians(real3* norm, real3* ptA, real3* ptB, int2* ids, real4* rot, real3* pos, real3* JXYZA, real3* JXYZB, real3* JUVWA, real3* JUVWB);
		void ComputeJacobians();

		void host_shurA(
				int2 *ids, bool *active, real *inv_mass, real3 *inv_inertia, real3 *JXYZA, real3 *JXYZB, real3 *JUVWA, real3 *JUVWB, real *gamma, real3 *updateV, real3 *updateO, real3* QXYZ, real3*QUVW,
				uint* offset);
		void host_shurB(
				int2 *ids, bool *active, real *inv_mass, real3 *inv_inertia, real * compliance, real * gamma, real3 *JXYZA, real3 *JXYZB, real3 *JUVWA, real3 *JUVWB, real3 *QXYZ, real3 *QUVW, real *AX);

		void host_Offsets(int2* ids_contacts, uint* Body);
		void host_Reduce_Shur(
				bool* active,
				real3* QXYZ,
				real3* QUVW,
				real *inv_mass,
				real3 *inv_inertia,
				real3* updateQXYZ,
				real3* updateQUVW,
				uint* d_body_num,
				uint* counter);
		void ShurA(custom_vector<real> &x);
		void ShurB(custom_vector<real> &x, custom_vector<real> & output);
	protected:

		custom_vector<real3> JXYZA_rigid_rigid;
		custom_vector<real3> JXYZB_rigid_rigid;
		custom_vector<real3> JUVWA_rigid_rigid;
		custom_vector<real3> JUVWB_rigid_rigid;
		custom_vector<real> comp_rigid_rigid;

		custom_vector<real3> vel_update, omg_update;

		custom_vector<uint> body_num;

		custom_vector<uint> update_number;
		custom_vector<uint> update_offset;
		custom_vector<uint> offset_counter;
		custom_vector<uint> body_number;
	}
				;
			}

#endif
