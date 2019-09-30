/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::normal_distribution;
using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 10;  // TODO: Set the number of particles
  particles = {};
  weights = {};
  
  std::default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
                                     
  for (int i = 0; i < num_particles; i++) {
    Particle p = {i, 
                  dist_x(gen),
                  dist_y(gen),
                  dist_theta(gen),
                  1, {}, {}, {}};
    particles.push_back(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
                                     
  for (auto& p : particles) {
    double theta, x, y;
    if (fabs(yaw_rate) > 1e-5) {
      theta = p.theta + yaw_rate * delta_t;
      x = p.x + velocity / yaw_rate * (sin(theta) - sin(p.theta));
      y = p.y + velocity / yaw_rate * (cos(p.theta) - cos(theta));
    } else {
      theta = p.theta;
      x = p.x + velocity * delta_t * cos(theta);
      y = p.y + velocity * delta_t * sin(theta);
    }
    
    normal_distribution<double> dist_x(x, std_pos[0]);
    normal_distribution<double> dist_y(y, std_pos[1]);
    normal_distribution<double> dist_theta(theta, std_pos[2]);
    
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (LandmarkObs& obs : observations) {
    double minDistance = std::numeric_limits<double>::max();
    int minDistanceId = -1;
    for (LandmarkObs& pred : predicted) {
      double distance = dist(pred.x, pred.y, obs.x, obs.y);
      if (distance < minDistance) {
        minDistance = distance;
        minDistanceId = pred.id;
      }
    }
    obs.id = minDistanceId;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  weights.clear();
  for (auto& p : particles) {
    // Map observations from car cs to particle cs
    vector<LandmarkObs> map_observations;
    for (auto& obs : observations) {
      auto x = p.x + cos(p.theta) * obs.x - sin(p.theta) * obs.y;
      auto y = p.y + sin(p.theta) * obs.x + cos(p.theta) * obs.y;
      map_observations.push_back(LandmarkObs{obs.id, x, y});
    }
    // Filter map landmarks within sensor range
    vector<LandmarkObs> predicted;
    for (auto& landmark : map_landmarks.landmark_list) {
      if (dist(p.x, p.y, landmark.x_f, landmark.y_f) <= sensor_range) {
        predicted.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
      }
    }
    
    dataAssociation(predicted, map_observations);
    // Calculate weight
    double weight = 1;
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;
    for (auto& obs : map_observations) {
      // Find landmark by id
      LandmarkObs lm;
      for (auto& pred : predicted) {
        if (obs.id == pred.id) {
          lm = pred;
          break;
        }
      }

      // Calculate probability density and update weight
      double exponent = - (pow(obs.x - lm.x, 2) / 2 / pow(std_landmark[0], 2) + pow(obs.y - lm.y, 2) / 2 / pow(std_landmark[1], 2));
      double p = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]) * exp(exponent);
      weight *= p;

      associations.push_back(obs.id);
      sense_x.push_back(obs.x);
      sense_y.push_back(obs.y);
    }
    p.weight = weight;
    weights.push_back(weight);
    SetAssociations(p, associations, sense_x, sense_y);
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::default_random_engine gen;
  std::discrete_distribution<> dist(weights.begin(), weights.end());

  vector<Particle> new_particles = {};
  for (int i = 0; i < num_particles; i++) {
    new_particles.push_back(particles[dist(gen)]);
  }

  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}