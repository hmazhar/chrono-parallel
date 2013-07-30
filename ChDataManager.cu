#include "ChDataManager.h"
using namespace chrono;
ChGPUDataManager::ChGPUDataManager() {
	number_of_contacts = 0;
	number_of_models = 0;
	number_of_objects = 0;
	number_of_bilaterals = 0;
	number_of_contacts_possible = 0;
	copyContacts = false;
}
ChGPUDataManager::~ChGPUDataManager() {
}
void ChGPUDataManager::Copy(GPUCOPYTYPE type) {
	CopyBodyData(type);
	CopyBilateralData(type);
	CopyGeometryData(type);
}

void ChGPUDataManager::CopyBodyData(GPUCOPYTYPE type) {
	if (type == HOST_TO_DEVICE) {
		gpu_data.device_vel_data = host_vel_data;
		gpu_data.device_omg_data = host_omg_data;
		gpu_data.device_pos_data = host_pos_data;
		gpu_data.device_rot_data = host_rot_data;
		gpu_data.device_inr_data = host_inr_data;
		gpu_data.device_acc_data = host_vel_data;
		gpu_data.device_active_data = host_active_data;
		gpu_data.device_mass_data = host_mass_data;
		gpu_data.device_fric_data = host_fric_data;
		gpu_data.device_cohesion_data = host_cohesion_data;
		gpu_data.device_lim_data = host_lim_data;
		gpu_data.device_frc_data = host_frc_data;
		gpu_data.device_trq_data = host_trq_data;
	} else {
		host_vel_data = gpu_data.device_vel_data;
		host_omg_data = gpu_data.device_omg_data;
		host_pos_data = gpu_data.device_pos_data;
		host_rot_data = gpu_data.device_rot_data;
		host_acc_data = gpu_data.device_acc_data;
		host_gyr_data = gpu_data.device_gyr_data;
	}

}

void ChGPUDataManager::CopyBilateralData(GPUCOPYTYPE type) {
	if (type == HOST_TO_DEVICE) {
		gpu_data.device_JXYZA_bilateral = host_JXYZA_bilateral;
		gpu_data.device_JXYZB_bilateral = host_JXYZB_bilateral;
		gpu_data.device_JUVWA_bilateral = host_JUVWA_bilateral;
		gpu_data.device_JUVWB_bilateral = host_JUVWB_bilateral;
		gpu_data.device_residual_bilateral = host_residual_bilateral;
		gpu_data.device_correction_bilateral = host_correction_bilateral;
		gpu_data.device_bids_bilateral = host_bids_bilateral;
		gpu_data.device_gamma_bilateral = host_gamma_bilateral;
	} else {
		host_gamma_bilateral = gpu_data.device_gamma_bilateral;
	}
}


void ChGPUDataManager::CopyGeometryData(GPUCOPYTYPE type) {
	if (type == HOST_TO_DEVICE) {
		gpu_data.device_typ_data = host_typ_data;
		gpu_data.device_ObA_data = host_ObA_data;
		gpu_data.device_ObB_data = host_ObB_data;
		gpu_data.device_ObC_data = host_ObC_data;
		gpu_data.device_ObR_data = host_ObR_data;
		gpu_data.device_fam_data = host_fam_data;
		gpu_data.device_id_data = host_id_data;
	}
}
void ChGPUDataManager::CopyContactData(GPUCOPYTYPE type) {
	if (type == HOST_TO_DEVICE) {
		gpu_data.device_norm_data = host_norm_data;
		gpu_data.device_cpta_data = host_cpta_data;
		gpu_data.device_cptb_data = host_cptb_data;
		gpu_data.device_dpth_data = host_dpth_data;
		gpu_data.device_bids_data = host_bids_data;
	} else {
		host_norm_data = gpu_data.device_norm_data;
		host_cpta_data = gpu_data.device_cpta_data;
		host_cptb_data = gpu_data.device_cptb_data;
		host_dpth_data = gpu_data.device_dpth_data;
		host_bids_data = gpu_data.device_bids_data;
		host_pair_data = gpu_data.device_pair_data;
	}
}

void ChGPUDataManager::DeviceToHostPairData() {
	host_pair_data = gpu_data.device_pair_data;
}
void ChGPUDataManager::DeviceToHostJacobians() {
	host_JXYZA_data = gpu_data.device_JXYZA_data;
	host_JUVWA_data = gpu_data.device_JUVWA_data;
	host_JXYZB_data = gpu_data.device_JXYZB_data;
	host_JUVWB_data = gpu_data.device_JUVWB_data;
	host_gam_data = gpu_data.device_gam_data;
}
