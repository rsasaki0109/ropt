#include <gtsam/geometry/Pose2.h>
#include <iostream>
#include <iomanip>
int main(){const gtsam::Vector3 d(1,2,.3);const gtsam::Pose2 origin;const auto p=origin.retract(d);std::cout<<std::setprecision(17)<<p.x()<<' '<<p.y()<<' '<<p.theta()<<'\n';return (gtsam::Vector3(p.x(),p.y(),p.theta())-d).norm()<1e-14?0:1;}
