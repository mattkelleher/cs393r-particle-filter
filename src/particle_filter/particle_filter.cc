//========================================================================
//  This software is free: you can redistribute it and/or modify
//  it under the terms of the GNU Lesser General Public License Version 3,
//  as published by the Free Software Foundation.
//
//  This software is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU Lesser General Public License for more details.
//
//  You should have received a copy of the GNU Lesser General Public License
//  Version 3 in the file COPYING that came with this distribution.
//  If not, see <http://www.gnu.org/licenses/>.
//========================================================================
/*!
\file    particle-filter.cc
\brief   Particle Filter Starter Code
\author  Joydeep Biswas, (C) 2019
*/
//========================================================================

#include <algorithm>
#include <cmath>
#include <iostream>
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "shared/math/geometry.h"
#include "shared/math/line2d.h"
#include "shared/math/math_util.h"
#include "shared/util/timer.h"

#include "config_reader/config_reader.h"
#include "particle_filter.h"

#include "vector_map/vector_map.h"

using geometry::line2f;
using std::cout;
using std::endl;
using std::string;
using std::swap;
using std::vector;
using Eigen::Vector2f;
using Eigen::Vector2i;
using vector_map::VectorMap;

DEFINE_double(num_particles, 50, "Number of particles");

namespace particle_filter {

config_reader::ConfigReader config_reader_({"config/particle_filter.lua"});

ParticleFilter::ParticleFilter() :
    prev_odom_loc_(0, 0),
    prev_odom_angle_(0),
    odom_initialized_(false) {}

float Navigation::_Distance(Vector2f p1, Vector2f p2) {
  return sqrt(pow(p1.x() - p2.x(), 2) + pow(p1.y() - p2.y(), 2));
}

void ParticleFilter::GetParticles(vector<Particle>* particles) const {
  *particles = particles_;
}

void ParticleFilter::GetPredictedPointCloud(const Vector2f& loc,
                                            const float angle,
                                            int num_ranges,
                                            float range_min,
                                            float range_max,
                                            float angle_min,
                                            float angle_max,
                                            vector<Vector2f>* scan_ptr) {
  vector<Vector2f>& scan = *scan_ptr;
  const float distance_base2lidar = 1; // Measure
  float phi;
  // Compute what the predicted point cloud would be, if the car was at the pose
  // loc, angle, with the sensor characteristics defined by the provided
  // parameters.
  // This is NOT the motion model predict step: it is the prediction of the
  // expected observations, to be used for the update step.

  float x_base2lidar = distance_base2lidar * cos(angle);
  float y_base2lidar = distance_base2lidar * sin(angle);

  // Note: The returned values must be set using the `scan` variable:
  scan.resize(num_ranges);    // Usually 109 scans, 108 + 1
  // Fill in the entries of scan using array writes, e.g. scan[i] = ...
  for (size_t i = 0; i < (scan.size()-1); ++i) {
    phi = (-135 + 270/(scan.size()-1)*i) * 2*pi()/180;

    line2f sim_line(loc.x()+x_base2lidar+range_min*cos(phi),
                    loc.y()+y_base2lidar+range_min*sin(phi),
                    loc.x()+x_base2lidar+range_max*cos(phi),
                    loc.y()+y_base2lidar+range_max*sin(phi));

    for (size_t n = 0; n < map_.lines.size(); ++n) {
      const line2f map_line = map_.lines[n];

      Vector2f intersection_point; // Return variable
      intersects = map_line.Intersection(sim_line, &intersection_point);
      if (intersects && _Distance(Vector2f (sim_line.p0.x(), sim_line.p0.y(), intersection_point.x(), intersection_point.y())
                      < _Distance(Vector2f (sim_line.p0.x(), sim_line.p0.y(), scan[i]) {
        scan[i] = _Distance(Vector2f(sim_line.p0.x(), sim_line.p0.y()), Vector2f(intersection_point.x(), intersection_point.y()));
      } else {
        scan[i] = _Distance(Vector2f(sim_line.p0.x(), sim_line.p0.y()), Vector2f(sim_line.p1.x(), sim_line.p1.y()));
      }
    }
  }
}

void ParticleFilter::Update(const vector<float>& ranges, // Laser scans
                            float range_min,
                            float range_max,
                            float angle_min,
                            float angle_max,
                            Particle* p_ptr) {
  // Implement the update step of the particle filter here.
  // You will have to use the `GetPredictedPointCloud` to predict the expected
  // observations for each particle, and assign weights to the particles based
  // on the observation likelihood computed by relating the observation to the
  // predicted point cloud.

  // Tunable parameters
  const float sigma_s = 0.04; // m, Variance of LIDAR, from datasheet
  float d_short = 1; // Tunable parameter
  float d_long = 1; // Tunable parameter
  float gamma = 1; // Tunable parameter

  //Initialize variables
  vector double log_weights;
  vector double weights;
  double log_weights_sum = 0;

  log_weights.resize(ranges.size());
  weights.resize(log_weights.size());

  for (size_t i = 0; i < log_weights.size(); ++i) {
    log_weights[i] = ranges.size() * (-log(sigma_s)-0.5*log(2*pi())+gamma);
  }

  int n = 0;
  int m = 0;
  for (const particle_filter::Particle& p: p_ptr) {
    n = 0;

    vector<Vector2f> predicted_scan;
    particle_filter_.GetPredictedPointCloud(
        p_ptr.loc,
        p_ptr.angle,
        ranges.size(),
        range_min,
        range_max,
        angle_min,
        angle_max,
        &predicted_scan);

    for (auto s: predicted_scan) {
      if (s < range_min || s > range_max) {
        // Do nothing
      } else if (s < range[n] - d_short) {
        log_weights[i] += -d_short^2/(2*sigma_s^2);
      } else if (s > range[n] + d_long) {
        log_weights[i] += -d_long^2/(2*sigma_s^2);
      } else {
        log_weights[i] += -(_Distance(s, range[n]))^2/(2*sigma_s^2);
      } 
      log_weights_sum += log_weights[i];
      ++n;
    }

    for (size_t i = 0; i < log_weights.size(); ++i) {
      log_weights[i] = log_weights[i]/log_weights_sum;
      weights[i] = exp(log_weights[i]);
    }

    p.weight = p.weight * weights[m];
    ++m;
  }
}

void ParticleFilter::Resample() {
  // Resample the particles, proportional to their weights.
  // The current particles are in the `particles_` variable. 
  // Create a variable to store the new particles, and when done, replace the
  // old set of particles:
  // vector<Particle> new_particles';
  // During resampling: 
  //    new_particles.push_back(...)
  // After resampling:
  // particles_ = new_particles;

  // You will need to use the uniform random number generator provided. For
  // example, to generate a random number between 0 and 1:
  float x = rng_.UniformRandom(0, 1);
  printf("Random number drawn from uniform distribution between 0 and 1: %f\n",
         x);
}

void ParticleFilter::ObserveLaser(const vector<float>& ranges,
                                  float range_min,
                                  float range_max,
                                  float angle_min,
                                  float angle_max) {
  // A new laser scan observation is available (in the laser frame)
  // Call the Update and Resample steps as necessary.
}

void ParticleFilter::Predict(const Vector2f& odom_loc,
                             const float odom_angle) {
  // Implement the predict step of the particle filter here.
  // A new odometry value is available (in the odom frame)
  // Implement the motion model predict step here, to propagate the particles
  // forward based on odometry.


  // You will need to use the Gaussian random number generator provided. For
  // example, to generate a random number from a Gaussian with mean 0, and
  // standard deviation 2:
  float x = rng_.Gaussian(0.0, 2.0);
  printf("Random number drawn from Gaussian distribution with 0 mean and "
         "standard deviation of 2 : %f\n", x);
}

void ParticleFilter::Initialize(const string& map_file,
                                const Vector2f& loc,
                                const float angle) {
  // The "set_pose" button on the GUI was clicked, or an initialization message
  // was received from the log. Initialize the particles accordingly, e.g. with
  // some distribution around the provided location and angle.
  map_.Load(map_file);
}

void ParticleFilter::GetLocation(Eigen::Vector2f* loc_ptr, 
                                 float* angle_ptr) const {
  Vector2f& loc = *loc_ptr;
  float& angle = *angle_ptr;
  // Compute the best estimate of the robot's location based on the current set
  // of particles. The computed values must be set to the `loc` and `angle`
  // variables to return them. Modify the following assignments:
  loc = Vector2f(0, 0);
  angle = 0;
}


}  // namespace particle_filter
