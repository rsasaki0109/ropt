#pragma once
#include "precision.hpp"
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/BetweenFactor.h>
#include <memory>
namespace precise {
class Between:public gtsam::BetweenFactor<gtsam::Pose2> {
  using Parent=gtsam::BetweenFactor<gtsam::Pose2>;
  Whitening whitening_;
public:
  Between(gtsam::Key a,gtsam::Key b,const gtsam::Pose2& z,const Whitening& w):Parent(a,b,z,gtsam::noiseModel::Unit::Create(3)),whitening_(w){}
  gtsam::Vector evaluateError(const gtsam::Pose2& a,const gtsam::Pose2& b,boost::optional<gtsam::Matrix&> ha=boost::none,boost::optional<gtsam::Matrix&> hb=boost::none) const override {
    const gtsam::Vector3 r=Parent::evaluateError(a,b,ha,hb);
    if(ha){whitening_.matrix(*ha);}
    if(hb){whitening_.matrix(*hb);}
    return whitening_.vector(r);
  }
  gtsam::NonlinearFactor::shared_ptr clone() const override {return gtsam::NonlinearFactor::shared_ptr(new Between(*this));}
  bool equals(const gtsam::NonlinearFactor& other,double tol=1e-9) const override {
    const auto* p=dynamic_cast<const Between*>(&other);
    return p&&Parent::equals(other,tol)&&(whitening_.information-p->whitening_.information).norm()<=tol;
  }
};
// Exact conditioning on the fixed root: same residual/chart as a BetweenFactor,
// rather than using PriorFactor's approximate identity chart Jacobian.
class RootBetween:public gtsam::NoiseModelFactor1<gtsam::Pose2> {
  using Parent=gtsam::NoiseModelFactor1<gtsam::Pose2>;
  gtsam::Pose2 root_;
  gtsam::BetweenFactor<gtsam::Pose2> raw_;
  std::shared_ptr<const Whitening> whitening_;
public:
  RootBetween(gtsam::Key key,const gtsam::Pose2& root,const gtsam::Pose2& z,const gtsam::Matrix3& info):
    Parent(needs_precision(info)?gtsam::SharedNoiseModel(gtsam::noiseModel::Unit::Create(3)):gtsam::SharedNoiseModel(gtsam::noiseModel::Gaussian::Information(info)),key),
    root_(root),raw_(0,key,z,gtsam::noiseModel::Unit::Create(3)),
    whitening_(needs_precision(info)?std::make_shared<Whitening>(info):nullptr){}
  gtsam::Vector evaluateError(const gtsam::Pose2& x,boost::optional<gtsam::Matrix&> h=boost::none) const override {
    const gtsam::Vector3 r=raw_.evaluateError(root_,x,boost::none,h);
    if(!whitening_)return r;
    if(h){whitening_->matrix(*h);}
    return whitening_->vector(r);
  }
  gtsam::NonlinearFactor::shared_ptr clone() const override {return gtsam::NonlinearFactor::shared_ptr(new RootBetween(*this));}
  bool equals(const gtsam::NonlinearFactor& other,double tol=1e-9) const override {
    const auto* p=dynamic_cast<const RootBetween*>(&other);
    return p&&Parent::equals(other,tol)&&root_.equals(p->root_,tol)&&raw_.equals(p->raw_,tol)&&
      bool(whitening_)==bool(p->whitening_)&&(!whitening_||(whitening_->information-p->whitening_->information).norm()<=tol);
  }
};
}
