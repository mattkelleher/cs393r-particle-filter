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
    odom_initialized_(false),
    num_particles_(50), //TODO tune
    update_count_(0) {}

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
  // Compute what the predicted point cloud would be, if the car was at the pose
  // loc, angle, with the sensor characteristics defined by the provided
  // parameters.
  // This is NOT the motion model predict step: it is the prediction of the
  // expected observations, to be used for the update step.

  // Note: The returned values must be set using the `scan` variable:
  scan.resize(num_ranges);
  // Fill in the entries of scan using array writes, e.g. scan[i] = ...
  for (size_t i = 0; i < scan.size(); ++i) {
    scan[i] = Vector2f(0, 0);
  }

  // The line segments in the map are stored in the `map_.lines` variable. You
  // can iterate through them as:
  for (size_t i = 0; i < map_.lines.size(); ++i) {
    const line2f map_line = map_.lines[i];
    // The line2f class has helper functions that will be useful.
    // You can create a new line segment instance as follows, for :
    line2f my_line(1, 2, 3, 4); // Line segment from (1,2) to (3.4).
    // Access the end points using `.p0` and `.p1` members:
    printf("P0: %f, %f P1: %f,%f\n", 
           my_line.p0.x(),
           my_line.p0.y(),
           my_line.p1.x(),
           my_line.p1.y());

    // Check for intersections:
    bool intersects = map_line.Intersects(my_line);
    // You can also simultaneously check for intersection, and return the point
    // of intersection:
    Vector2f intersection_point; // Return variable
    intersects = map_line.Intersection(my_line, &intersection_point);
    if (intersects) {
      printf("Intersects at %f,%f\n", 
             intersection_point.x(),
             intersection_point.y());
    } else {
      printf("No intersection\n");
    }
  }
}


void ParticleFilter::Update(const vector<float>& ranges,
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
}

void ParticleFilter::Resample() {
  float cumulative_weight [50];
  vector<Particle> new_particles;
  int index_counter1 = 0;

  for(Particle i: particles_){
    if(index_counter1 == 0){
      cumulative_weight[index_counter1] = i.weight;
    } else {
      cumulative_weight[index_counter1] = i.weight + cumulative_weight[index_counter1-1];
    }
    index_counter1 ++; 
  }
  for(Particle i : particles_){
    float x = rng_.UniformRandom(0, 1);
    float rand;
    //not sure about the normalized weight
    rand = x * cumulative_weight[49];
    int index_counter2 = 0;
    for(float j : cumulative_weight){
      if(rand <= j){
	break;
      }
      index_counter2 ++;
    }
    new_particles.push_back(particles_[index_counter2]);
  }
  particles_ = new_particles;
  for(int i = 0; i < 50; i = i + 1){
    particles_[i].weight = 1/50;
  }
}


void ParticleFilter::ObserveLaser(const vector<float>& ranges,
                                  float range_min,
                                  float range_max,
                                  float angle_min,
                                  float angle_max) {
  // A new laser scan observation is available (in the laser frame)
  // Call the Update and Resample steps as necessary.
  int resample_frequency = 10;  //TODO tune

  for(auto p : particles_) {
    Update(ranges, range_min, range_max, angle_min, angle_max, &p);
  }
  
  if(update_count_ == resample_frequency) {
    update_count_ = 0;
    Resample();
  } 
}


void ParticleFilter::Predict(const Vector2f& odom_loc,
                             const float odom_angle) {
  // Implement the predict step of the particle filter here.
  // A new odometry value is available (in the odom frame)
  // Implement the motion model predict step here, to propagate the particles
  // forward based on odometry.
  float k1 = 1;  //TODO tune
  float k2 = 0.5;
  float k3 = 0.5; 
  float k4 = 1;

  Vector2f delta_loc  = odom_loc - prev_odom_loc_;
  float delta_angle = odom_angle - prev_odom_angle_;
  float std_loc = k1 * sqrt(pow(delta_loc.x(), 2) + pow(delta_loc.y(), 2)) + k2 * abs(delta_angle);
  float std_angle = k3 * sqrt(pow(delta_loc.x(), 2) + pow(delta_loc.y(), 2)) + k4 * abs(delta_angle);

  prev_odom_loc_ = odom_loc;
  prev_odom_angle_ = odom_angle;
  
  for(auto p : particles_) {
    p.loc.x() = p.loc.x() + delta_loc.x() + rng_.Gaussian(0.0, std_loc);
    p.loc.y() = p.loc.y() + delta_loc.y() + rng_.Gaussian(0.0, std_loc);
    p.angle = p.angle + delta_angle + rng_.Gaussian(0.0, std_angle);
  }
}


void ParticleFilter::Initialize(const string& map_file,
                                const Vector2f& loc,
                                const float angle) {
  // The "set_pose" button on the GUI was clicked, or an initialization message
  // was received from the log. Initialize the particles accordingly, e.g. with
  // some distribution around the provided location and angle.
  map_.Load(map_file);
  for(int i = 0; i < num_particles_; i++) {  // TODO most basic initalization, all particles start on top of 'initalized' location
    Particle p;
    //following are the randomized loc and angle, picking random points in the circle that is centered at the p.loc
    //float radius_rand = rng_.UniformRandom(0, 1);
    //float angle_rand = rng_.UniformRandom(0, 1);
    //p.loc.x = loc.x + radius * cos(angle_rand * 2 * PI);
    //p.loc.y = loc.y + radius * sin(angle_rand * 2 * PI);
    p.loc = loc;
    //float angle = rng_.UniformRandom(0, 1);
    //p.angle = 2 * PI * angle;
    p.angle = angle;
    p.weight = 0.0;
    particles_.push_back(p);
  }
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
