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
    last_update_loc_(0,0),
    num_particles_(50), //TODO tune
    scan_density_(10), 
    update_count_(0) {}

float ParticleFilter::_Distance(Vector2f p1, Vector2f p2) {
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
                                            vector<Vector2f>* scan_ptr) 
{
  vector<Vector2f>& scan = *scan_ptr;
  const float distance_base2lidar = 0.2; // From assignment 1
  float phi;
  // Compute what the predicted point cloud would be, if the car was at the pose
  // loc, angle, with the sensor characteristics defined by the provided
  // parameters.
  // This is NOT the motion model predict step: it is the prediction of the
  // expected observations, to be used for the update step.

  float x_base2lidar = distance_base2lidar * cos(angle);
  float y_base2lidar = distance_base2lidar * sin(angle);
  Vector2f laser_loc(loc.x() + x_base2lidar, loc.y() + y_base2lidar);
   
  // Note: The returned values must be set using the `scan` variable:
  scan.resize(num_ranges / scan_density_);    // Usually 109 scans, 108 + 1
  float increment = (angle_max - angle_min) / scan.size();
  // Fill in the entries of scan using array writes, e.g. scan[i] = ...
  for (size_t i = 0; i < scan.size(); i++) {
    phi = angle + (angle_min + i * increment);
    line2f sim_line(Vector2f(laser_loc.x() + range_min * cos(phi), laser_loc.y() + range_min * sin(phi)),
                    Vector2f(laser_loc.x() + range_max * cos(phi), laser_loc.y() + range_max * sin(phi))
                    );

    scan[i] = sim_line.p1;
    if(i == 1) {
      std::cout << "*************Laser loc: (" << laser_loc.x() << ", " << laser_loc.y() <<  ")   scan size: " << scan.size() << std::endl; 
      std::cout << "*************Sim line 0 (" << sim_line.p0.x() << ", " << sim_line.p0.y() <<  ")" << std::endl; 
      std::cout << "*************Sim line 1 (" << sim_line.p1.x() << ", " << sim_line.p1.y() <<  ")" << std::endl; 
      std::cout << "*************s[" << i << "] loc: (" << scan[i].x() << ", " << scan[i].y() <<  ")" << std::endl; 
      std::cout << "*************Laser loc to s init Dist: " << _Distance(laser_loc, scan[i]) <<  "/" << range_max << std::endl; 
    }
    for (size_t n = 0; n < map_.lines.size(); n++) {
      const line2f map_line = map_.lines[n];

      Vector2f intersection_point; // Return variable
      bool intersects = map_line.Intersection(sim_line, &intersection_point);
      if (intersects && (_Distance(Vector2f(sim_line.p0.x(), sim_line.p0.y()), Vector2f(intersection_point.x(), intersection_point.y()))
                      < _Distance(Vector2f(sim_line.p0.x(), sim_line.p0.y()), scan[i]))) { //TODO come back to double check logic 
                                                                                         //    do we need to worry about base frame vs frame of laser? (slightly offset) 
        scan[i] = intersection_point;
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
  const float sigma_s = 0.10; // m, Variance of LIDAR, from datasheet (0.04) + overestimation for robustness
  float d_short = 1; // Tunable parameter
  float d_long = 1; // Tunable parameter
  float gamma = 1; // Tunable parameter
  std::cout << "Entering Update " << std::endl;
  //Initialize variables
  vector<double>log_weights;
  vector<double>weights;
  double log_weights_max = 0;

  log_weights.resize(particles_.size());
  weights.resize(log_weights.size());

  for (size_t i = 0; i < log_weights.size(); ++i) {
    log_weights[i] = ranges.size() / scan_density_ * (-log(sigma_s)-0.5*log(2*M_PI) + gamma);
  }

  int m = 0; // particle index
  int n = 0; // predicted_scan index
  
  for (Particle p: particles_) {

    vector<Vector2f> predicted_scan;
    GetPredictedPointCloud(
        p.loc,
        p.angle, 
        ranges.size(), 
        range_min,
        range_max,
        angle_min,
        angle_max,
        &predicted_scan
    );
    Vector2f laser_loc;
    laser_loc.x() = p.loc.x() + 0.2 * cos(M_PI / 180 * p.angle); //0.2 is distance from base frame to laser
    laser_loc.y() = p.loc.y() + 0.2 * sin(M_PI / 180 * p.angle);
   
    n = 0;
    for (auto s: predicted_scan) {
      double dist = _Distance(laser_loc, s);
      if (dist < range_min || dist > range_max) {
        // Do nothing
      } else if (dist < ranges[n*scan_density_] - d_short) {
        log_weights[m] += -1 * pow(d_short,2) / (2 * pow(sigma_s,2));
      } else if (dist > ranges[n*scan_density_] + d_long) {
        log_weights[m] += -1 * pow(d_long, 2) / (2 * pow(sigma_s,2));
      } else {
        log_weights[m] += -1 * pow(dist - ranges[n*scan_density_], 2) / (2 * pow(sigma_s,2));
      } 
      n++;
    }
    if(m == 1) {
      std::cout << "*****Log weight: " << log_weights[m] << std::endl;
      std::cout << "*****Start weight: " << particles_[0].weight << std::endl;
    }
    if(abs(log_weights[m]) > log_weights_max) {
       log_weights_max = log_weights[m];
       std::cout << "***** NEW MAX !!: " << log_weights_max << std::endl;
    } 
    m++;
 }  


  // Normalizing log weights and set weight value
  double weight_sum = 0;
  std::cout << "MAX LOG WEIGHT: " << log_weights_max << std::endl;
  for (unsigned int i = 0; i < log_weights.size(); i++) {
    log_weights[i] = log_weights[i] - log_weights_max;
    weights[i] = exp(log_weights[i]);
    weights[i] = particles_[i].weight * weights[i];
    weight_sum += weights[i];
  }
  for(size_t i = 0; i < weights.size(); i++ ) {
    particles_[i].weight = weights[i] / weight_sum;
    if(i == 0) {    
      std::cout << "***** Normalized Log weight: " << log_weights[i] << std::endl;
      std::cout << "*****END weight: " << particles_[0].weight << std::endl;
    }
  }
} 

void ParticleFilter::Resample() {
  float cumulative_weight [num_particles_];
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
  float rand = rng_.UniformRandom(0, 1);
  for(size_t i = 0; i < particles_.size(); i++){
    rand = rand + float(i) / num_particles_;
    if(rand > 1) {
      rand = rand - 1;
    } 
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
  for(size_t i = 0; i < particles_.size(); i = i + 1){
    particles_[i].weight = 1.0/num_particles_;
  }
  std::cout << "Exiting resample" << std::endl;
}


void ParticleFilter::ObserveLaser(const vector<float>& ranges,
                                  float range_min,
                                  float range_max,
                                  float angle_min,
                                  float angle_max) {
  // A new laser scan observation is available (in the laser frame)
  // Call the Update and Resample steps as necessary.
  int resample_frequency = 10;  //TODO tune

  std::cout << "Entering Observe Laser" << std::endl;
  if(_Distance(prev_odom_loc_, last_update_loc_) > 0.03) {
    last_update_loc_ = prev_odom_loc_;
    Update(ranges, range_min, range_max, angle_min, angle_max, &particles_[1]);
    update_count_++;
  
    if(update_count_ == resample_frequency) {
      update_count_ = 0;
      Resample();
    }
  } 
}


void ParticleFilter::Predict(const Vector2f& odom_loc,
                             const float odom_angle) {
  // Implement the predict step of the particle filter here.
  // A new odometry value is available (in the odom frame)
  // Implement the motion model predict step here, to propagate the particles
  // forward based on odometry.
  
  //float k1 = 0.001;  //TODO tune
  //float k2 = 0.0005;
  //float k3 = 0.0005; 
  //float k4 = 0.001;

  std::cout << "Entering Predict" << std::endl;
  Vector2f delta_loc(0,0);
  delta_loc.x()  = abs(odom_loc.x() - prev_odom_loc_.x());
  delta_loc.y()  = abs(odom_loc.y() - prev_odom_loc_.y());

  float delta_angle = odom_angle - prev_odom_angle_;
  //float std_loc = k1 * sqrt(pow(delta_loc.x(), 2) + pow(delta_loc.y(), 2)) + k2 * abs(delta_angle);
  //float std_angle = k3 * sqrt(pow(delta_loc.x(), 2) + pow(delta_loc.y(), 2)) + k4 * abs(delta_angle);

  std::cout << "Delta Loc: (" << delta_loc.x() << ", " << delta_loc.y() << ")" << std::endl;
  std::cout << "Odom Loc: (" << odom_loc.x() << ", " << odom_loc.y() << ")" << std::endl;
  std::cout << "prev_odom Loc: (" << prev_odom_loc_.x() << ", " << prev_odom_loc_.y() << ")" << std::endl;
  
  for(size_t i = 0; i < particles_.size(); i++) {
    if(i == 0){
      std::cout << "OLD P Loc: (" << particles_[i].loc.x() << ", " << particles_[i].loc.y() << ") Angle: " << particles_[i].angle << std::endl;
    } 
    particles_[i].loc.x() += delta_loc.x(); //+ rng_.Gaussian(0.0, std_loc);
    particles_[i].loc.y() += delta_loc.y(); //+ rng_.Gaussian(0.0, std_loc);
    particles_[i].angle += delta_angle; // + rng_.Gaussian(0.0, std_angle);
    if(i == 0){
      std::cout << "P Loc: (" << particles_[i].loc.x() << ", " << particles_[i].loc.y() << ") Angle: " << particles_[i].angle << std::endl;
    } 
  }
  prev_odom_loc_.x() = odom_loc.x();
  prev_odom_loc_.y() = odom_loc.y();
  prev_odom_angle_ = odom_angle;
}


void ParticleFilter::Initialize(const string& map_file,
                                const Vector2f& loc,
                                const float angle) {
  // The "set_pose" button on the GUI was clicked, or an initialization message
  // was received from the log. Initialize the particles accordingly, e.g. with
  // some distribution around the provided location and angle.
  std::cout << "Entering Initialize function!" << std::endl;
  map_.Load(map_file);
  
  last_update_loc_ = loc;
  prev_odom_loc_ = loc;
  prev_odom_angle_ = angle;
 
  // Remove all previous particles
  std::cout << "Removing old particles" << std::endl; 
  while(particles_.size() > 0) {
    particles_.pop_back();
  }
  
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
    p.weight = 1.0 / num_particles_;
    std::cout << "Creating particle: " << i << " at location: " << p.loc.x() << ", " << p.loc.y() << " with angel: " << p.angle << " and weight: " << p.weight << std::endl;
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
  float x = 0;
  float y = 0;
  float theta_x = 0;
  float theta_y = 0;
  for(auto p : particles_) {
    x = x + p.weight * p.loc.x();
    y = y + p.weight * p.loc.y();
    theta_x = theta_x + p.weight * cos(M_PI / 180 * p.angle);
    theta_y = theta_y + p.weight * sin(M_PI / 180 * p.angle); 
  } 

  loc = Vector2f(x, y);
  angle = 180.0 / M_PI * atan2(theta_y, theta_x);
  std::cout << "Location: (" << loc.x() << ", " << loc.y() << ") Angle: " << angle << std::endl;
}


}  // namespace particle_filter
