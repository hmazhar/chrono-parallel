#include "generators.h"

namespace chrono {
namespace utils {


// ===============================================================================
// Implementation of MixtureIngredient class
// ===============================================================================

// Constructor: create a new mixture ingredient for the specified generator, of a
// given type and with a specified ratio in the final mixture. This constructor is
// private as a new mixture ingredient can only be created through a generator.
MixtureIngredient::MixtureIngredient(Generator*  generator,
                                     MixtureType type,
                                     double      ratio)
:	m_generator(generator),
	m_type(type),
	m_ratio(ratio),
	m_cumRatio(0),
	m_defMaterialDVI(new ChMaterialSurface),
	m_defMaterialDEM(new ChMaterialSurfaceDEM),
	m_defDensity(1),
	m_defSize(ChVector<>(1, 1, 1)),
	m_frictionDist(NULL),
	m_cohesionDist(NULL),
	m_youngDist(NULL),
	m_poissonDist(NULL),
	m_dissipationDist(NULL),
	m_densityDist(NULL),
	m_sizeDist(NULL)
{
}

// Destructor:: free the various distribution associated with this ingredient
MixtureIngredient::~MixtureIngredient()
{
	freeMaterialDist();
	delete m_densityDist;
	delete m_sizeDist;
}

// Functions to set constant properties for all objects created based on this
// ingredient.
void
MixtureIngredient::setDefaultMaterialDVI(const ChSharedPtr<ChMaterialSurface>& mat)
{
	m_defMaterialDVI = mat;
	freeMaterialDist();
}

void
MixtureIngredient::setDefaultMaterialDEM(const ChSharedPtr<ChMaterialSurfaceDEM>& mat)
{
	m_defMaterialDEM = mat;
	freeMaterialDist();
}

void
MixtureIngredient::setDefaultDensity(double density)
{
	m_defDensity = density;
	delete m_densityDist;
}

void
MixtureIngredient::setDefaultSize(const ChVector<>& size)
{
	m_defSize = size;
	delete m_sizeDist;
}

// Functions to set parameters of truncated normal distributions for the various
// attributes of an ingredient.  Properties of objects created based on this 
// ingredient will be obtained by drawing from the corresponding distribution.
// If no distribution is specified for a particular property, the default constant
// value is used.
void
MixtureIngredient::setDistributionFriction(float friction_mean, float friction_stddev, float friction_min, float friction_max)
{
	m_frictionDist = new std::normal_distribution<float>(friction_mean, friction_stddev);
	m_minFriction = friction_min;
	m_maxFriction = friction_max;
}

void
MixtureIngredient::setDistributionCohesion(float cohesion_mean, float cohesion_stddev, float cohesion_min, float cohesion_max)
{
	m_cohesionDist = new std::normal_distribution<float>(cohesion_mean, cohesion_stddev);
	m_minCohesion = cohesion_min;
	m_maxCohesion = cohesion_max;
}

void
MixtureIngredient::setDistributionYoung(float young_mean, float young_stddev, float young_min, float young_max)
{
	m_youngDist = new std::normal_distribution<float>(young_mean, young_stddev);
	m_minYoung = young_min;
	m_maxYoung = young_max;
}

void
MixtureIngredient::setDistributionPoisson(float poisson_mean, float poisson_stddev, float poisson_min, float poisson_max)
{
	m_poissonDist = new std::normal_distribution<float>(poisson_mean, poisson_stddev);
	m_minPoisson = poisson_min;
	m_maxPoisson = poisson_max;
}

void
MixtureIngredient::setDistributionDissipation(float dissipation_mean, float dissipation_stddev, float dissipation_min, float dissipation_max)
{
	m_dissipationDist = new std::normal_distribution<float>(dissipation_mean, dissipation_stddev);
	m_minDissipation = dissipation_min;
	m_maxDissipation = dissipation_max;
}

void
MixtureIngredient::setDistributionDensity(double density_mean, double density_stddev, double density_min, double density_max)
{
	m_densityDist = new std::normal_distribution<double>(density_mean, density_stddev);
	m_minDensity = density_min;
	m_maxDensity = density_max;
}

void
MixtureIngredient::setDistributionSize(double size_mean, double size_stddev, const ChVector<>& size_min, const ChVector<>& size_max)
{
	m_sizeDist = new std::normal_distribution<double>(size_mean, size_stddev);
	m_minSize = size_min;
	m_maxSize = size_max;
}

// Utility function to delete all distributions associated with this ingredient.
void
MixtureIngredient::freeMaterialDist()
{
	delete m_frictionDist;
	delete m_cohesionDist;
	delete m_youngDist;
	delete m_poissonDist;
	delete m_dissipationDist;
}

// Modify the specified DVI material surface based on attributes of this ingredient.
void
MixtureIngredient::setMaterialProperties(ChSharedPtr<ChMaterialSurface>& mat)
{
	if (m_frictionDist)
		mat->SetFriction(sampleTruncatedDist<float>(*m_frictionDist, m_minFriction, m_maxFriction));
	else
		mat->SetFriction(m_defMaterialDVI->GetSfriction());

	if (m_cohesionDist)
		mat->SetCohesion(sampleTruncatedDist<float>(*m_cohesionDist, m_minCohesion, m_maxCohesion));
	else
		mat->SetCohesion(m_defMaterialDVI->GetCohesion());
}

// Modify the specified DEM material surface based on attributes of this ingredient.
void
MixtureIngredient::setMaterialProperties(ChSharedPtr<ChMaterialSurfaceDEM>& mat)
{
	if (m_youngDist)
		mat->SetYoungModulus(sampleTruncatedDist<float>(*m_youngDist, m_minYoung, m_maxYoung));
	else
		mat->SetYoungModulus(m_defMaterialDEM->GetYoungModulus());

	if (m_poissonDist)
		mat->SetPoissonRatio(sampleTruncatedDist<float>(*m_poissonDist, m_minPoisson, m_maxPoisson));
	else
		mat->SetPoissonRatio(m_defMaterialDEM->GetPoissonRatio());

	if (m_frictionDist)
		mat->SetFriction(sampleTruncatedDist<float>(*m_frictionDist, m_minFriction, m_maxFriction));
	else
		mat->SetFriction(m_defMaterialDEM->GetSfriction());

	if (m_dissipationDist)
		mat->SetDissipationFactor(sampleTruncatedDist<float>(*m_dissipationDist, m_minDissipation, m_maxDissipation));
	else
		mat->SetDissipationFactor(m_defMaterialDEM->GetDissipationFactor());

	if (m_cohesionDist)
		mat->SetCohesion(sampleTruncatedDist<float>(*m_cohesionDist, m_minCohesion, m_maxCohesion));
	else
		mat->SetCohesion(m_defMaterialDEM->GetCohesion());
}

// Return a size for an object created based on attributes of this ingredient.
ChVector<>
MixtureIngredient::getSize()
{
	if (m_sizeDist)
		return ChVector<>(sampleTruncatedDist<double>(*m_sizeDist, m_minSize.x, m_maxSize.x),
		                  sampleTruncatedDist<double>(*m_sizeDist, m_minSize.y, m_maxSize.y),
		                  sampleTruncatedDist<double>(*m_sizeDist, m_minSize.z, m_maxSize.z));

	return m_defSize;
}

// Return a density for an object created based on attributes of this ingredient.
double
MixtureIngredient::getDensity()
{
	if (m_densityDist)
		return sampleTruncatedDist<double>(*m_densityDist, m_minDensity, m_maxDensity);

	return m_defDensity;
}

// Calculate the volume and gyration tensor of an object of given size created
// from this ingredient type.
void
MixtureIngredient::calcGeometricProps(const ChVector<>& size, double& volume, ChVector<>& gyration)
{
	switch (m_type) {
	case SPHERE:
		volume = 4.0/3 * Pi * size.x * size.x * size.x;
		gyration = 2.0/5 * size.x * size.x * ChVector<>(1,1,1);
		break;
	case ELLIPSOID:
		volume = 4.0/3 * Pi * size.x * size.y * size.z;
		gyration = 1.0/5 * ChVector<>(size.y * size.y + size.z * size.z,
		                            size.x * size.x + size.z * size.z,
		                            size.x * size.x + size.y * size.y);
		break;
	case BOX:
		volume = 8 * size.x * size.y * size.z;
		gyration = 1.0/12 * ChVector<>(size.y * size.y + size.z * size.z,
		                               size.x * size.x + size.z * size.z,
		                               size.x * size.x + size.y * size.y);
		break;
	}
}

// Calculate a necessary minimum separation based on the largest possible
// dimension of an object created based on attributes of this ingredient.
double
MixtureIngredient::calcMinSeparation()
{
	if (m_sizeDist)
		return std::max(m_maxSize.x, std::max(m_maxSize.y, m_maxSize.z));

	return std::max(m_defSize.x, std::max(m_defSize.y, m_defSize.z));
}


// ===============================================================================
// Implementation of Generator class
// ===============================================================================

// Constructor: create a generator for the specified system.
Generator::Generator(ChSystem* system)
:	m_system(system),
	m_mixDist(0,1),
	m_crtBodyId(0),
	m_totalNumBodies(0),
	m_totalMass(0),
	m_totalVolume(0)
{
	m_sysType = GetSystemType(system);
}

// Destructor
Generator::~Generator()
{
}

// Add a new ingredient to the current mixture by specifying its type
// and the ratio in the final mixture. A smart pointer to the new
// mixture ingredient is returned to allow modifying its properties.
MixtureIngredientPtr&
Generator::AddMixtureIngredient(MixtureType type,
                                double      ratio)
{
	MixtureIngredientPtr ingredient(new MixtureIngredient(this, type, ratio));
	m_mixture.push_back(ingredient);
	
	return m_mixture.back();
}

// Create objects in a box domain using the specified type of point
// sampler and separation distance and the current mixture settings.
// The types of objects created are selected randomly with probability
// proportional to the ratio of that ingredient in the mixture.
void Generator::createObjectsBox(SamplingType      sType,
                                 double            dist,
                                 const ChVector<>& pos,
                                 const ChVector<>& hdims,
                                 const ChVector<>& vel)
{
	// Normalize the mixture ratios
	normalizeMixture();

	// Generate the object locations
	if (m_sysType == SEQUENTIAL_DEM || m_sysType == PARALLEL_DEM)
		dist = calcMinSeparation(dist);

	PointVector points;
	switch (sType) {
	case REGULAR_GRID:
		{
		GridSampler<> sampler(dist);
		points = sampler.SampleBox(pos, hdims);
		}
		break;
	case POISSON_DISK:
		{
		PDSampler<> sampler(dist);
		points = sampler.SampleBox(pos, hdims);
		}
		break;
	}

	createObjects(points, vel);
}


// Create objects in a cylindrical domain using the specified type of point
// sampler and separation distance and the current mixture settings.
// The types of objects created are selected randomly with probability
// proportional to the ratio of that ingredient in the mixture.
void Generator::createObjectsCylinderX(SamplingType sType, double dist,
                                       const ChVector<>& pos, float radius, float halfHeight,
                                       const ChVector<>& vel)
{
	// Normalize the mixture ratios
	normalizeMixture();

	// Generate the object locations
	if (m_sysType == SEQUENTIAL_DEM || m_sysType == PARALLEL_DEM)
		dist = calcMinSeparation(dist);

	PointVector points;
	switch (sType) {
	case REGULAR_GRID:
		{
		GridSampler<> sampler(dist);
		points = sampler.SampleCylinderX(pos, radius, halfHeight);
		}
		break;
	case POISSON_DISK:
		{
		PDSampler<> sampler(dist);
		points = sampler.SampleCylinderX(pos, radius, halfHeight);
		}
		break;
	}

	createObjects(points, vel);
}

void Generator::createObjectsCylinderY(SamplingType sType, double dist,
                                       const ChVector<>& pos, float radius, float halfHeight,
                                       const ChVector<>& vel)
{
	// Normalize the mixture ratios
	normalizeMixture();

	// Generate the object locations
	if (m_sysType == SEQUENTIAL_DEM || m_sysType == PARALLEL_DEM)
		dist = calcMinSeparation(dist);

	PointVector points;
	switch (sType) {
	case REGULAR_GRID:
		{
		GridSampler<> sampler(dist);
		points = sampler.SampleCylinderY(pos, radius, halfHeight);
		}
		break;
	case POISSON_DISK:
		{
		PDSampler<> sampler(dist);
		points = sampler.SampleCylinderY(pos, radius, halfHeight);
		}
		break;
	}

	createObjects(points, vel);
}

void Generator::createObjectsCylinderZ(SamplingType sType, double dist,
                                       const ChVector<>& pos, float radius, float halfHeight,
                                       const ChVector<>& vel)
{
	// Normalize the mixture ratios
	normalizeMixture();

	// Generate the object locations
	if (m_sysType == SEQUENTIAL_DEM || m_sysType == PARALLEL_DEM)
		dist = calcMinSeparation(dist);

	PointVector points;
	switch (sType) {
	case REGULAR_GRID:
		{
		GridSampler<> sampler(dist);
		points = sampler.SampleCylinderZ(pos, radius, halfHeight);
		}
		break;
	case POISSON_DISK:
		{
		PDSampler<> sampler(dist);
		points = sampler.SampleCylinderZ(pos, radius, halfHeight);
		}
		break;
	}

	createObjects(points, vel);
}


// Normalize the mixture ratios (so that their sum is 1) and calculate
// the exclusive scan of these ratios.
void Generator::normalizeMixture()
{
	if (m_mixture.empty()) {
		AddMixtureIngredient(SPHERE, 1);
		return;
	}

	double sum = 0;

	for (int i = 0; i < m_mixture.size(); i++)
		sum += m_mixture[i]->m_ratio;

	for (int i = 0; i < m_mixture.size(); i++) {
		m_mixture[i]->m_ratio /= sum;
		if (i == 0)
			m_mixture[i]->m_cumRatio = 0;
		else
			m_mixture[i]->m_cumRatio = m_mixture[i-1]->m_cumRatio + m_mixture[i-1]->m_ratio;
	}
}

// Select one of the mixture types with probability proportional to that
// type's ratio in the mixture.
int Generator::selectIngredient()
{
	double val = m_mixDist(rengine());

	for (int i = m_mixture.size() - 1; i >= 0; i--) {
		if (val > m_mixture[i]->m_cumRatio)
			return i;
	}

	return 0;
}

// Calculate the minimum distance to ensure contact shapes never overlap.
double Generator::calcMinSeparation(double sep)
{
	for (int i = 0; i < m_mixture.size(); i++)
		sep = std::max(sep, m_mixture[i]->calcMinSeparation());

	return sep;
}

// Create objects at the specified locations using the current mixture settings.
void Generator::createObjects(const PointVector& points,
                              const ChVector<>&  vel)
{
	for (int i = 0; i < points.size(); i++) {

		// Select the type of object to be created.
		int index = selectIngredient();

		// Create the body and set contact material
		ChBody* body;

		switch (m_sysType) {
		case SEQUENTIAL_DVI:
			body = new ChBody();
			m_mixture[index]->setMaterialProperties(body->GetMaterialSurface());
			break;
		case SEQUENTIAL_DEM:
			body = new ChBodyDEM();
			m_mixture[index]->setMaterialProperties(((ChBodyDEM*) body)->GetMaterialSurfaceDEM());
			break;
		case PARALLEL_DVI:
			body = new ChBody(new ChCollisionModelParallel);
			m_mixture[index]->setMaterialProperties(body->GetMaterialSurface());
			break;
		case PARALLEL_DEM:
			body = new ChBodyDEM(new ChCollisionModelParallel);
			m_mixture[index]->setMaterialProperties(((ChBodyDEM*) body)->GetMaterialSurfaceDEM());
			break;
		}

		// Set identifier
		body->SetIdentifier(m_crtBodyId++);

		// Set position and orientation
		body->SetPos(points[i]);
		body->SetRot(ChQuaternion<>(1, 0, 0, 0));
		body->SetPos_dt(vel);
		body->SetBodyFixed(false);
		body->SetCollide(true);

		// Get size and density; calculate geometric properties
		ChVector<> size = m_mixture[index]->getSize();
		double     density = m_mixture[index]->getDensity();
		double     volume;
		ChVector<> gyration;
		m_mixture[index]->calcGeometricProps(size, volume, gyration);
		double     mass = density * volume;

		// Set mass properties
		body->SetMass(mass);
		body->SetInertiaXX(mass * gyration);
		m_totalMass += mass;
		m_totalVolume += volume;

		// Add collision geometry
		body->GetCollisionModel()->ClearModel();

		switch (m_mixture[index]->m_type) {
		case SPHERE:
			AddSphereGeometry(body, size.x);
			break;
		case ELLIPSOID:
			AddEllipsoidGeometry(body, size);
			break;
		case BOX:
			AddBoxGeometry(body, size);
			break;
		}

		body->GetCollisionModel()->BuildModel();

		// Attach the body to the system and append to list of generated bodies.
		ChSharedPtr<ChBody>  bodyPtr(body);

		switch (m_sysType) {
		case SEQUENTIAL_DVI:
		case SEQUENTIAL_DEM:
			m_system->AddBody(bodyPtr);
			break;
		case PARALLEL_DVI:
		case PARALLEL_DEM:
			((ChSystemParallel*) m_system)->AddBody(bodyPtr);
			break;
		}

		m_bodies.push_back(BodyInfo(m_mixture[index]->m_type, density, size, bodyPtr));
	}

	m_totalNumBodies += points.size();
}

// Write body information to a CSV file
void
Generator::writeObjectInfo(const std::string& filename)
{
	CSV_writer csv;

	for (int i = 0; i < m_bodies.size(); i++) {
		csv << m_bodies[i].m_type;
		csv << m_bodies[i].m_body->GetPos() << m_bodies[i].m_size;
		csv << m_bodies[i].m_density << m_bodies[i].m_body->GetMass();

		switch (m_sysType) {
		case SEQUENTIAL_DVI:
		case PARALLEL_DVI:
			{
				ChSharedPtr<ChMaterialSurface> mat = m_bodies[i].m_body->GetMaterialSurface();
				csv << mat->GetSfriction() << mat->GetCohesion();
			}
			break;
		case SEQUENTIAL_DEM:
		case PARALLEL_DEM:
			{
				ChSharedPtr<ChMaterialSurfaceDEM> mat = ((ChBodyDEM*) m_bodies[i].m_body.get_ptr())->GetMaterialSurfaceDEM();
				csv << mat->GetYoungModulus() << mat->GetPoissonRatio() << mat->GetSfriction() << mat->GetDissipationFactor();
			}
			break;
		}

		csv << std::endl;
	}

	csv.write_to_file(filename);
}


}  // namespace utils
}  // namespace chrono