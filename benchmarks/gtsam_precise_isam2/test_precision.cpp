#include "factors.hpp"
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <iostream>
using namespace gtsam;
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
using Big=boost::multiprecision::number<boost::multiprecision::cpp_bin_float<200>>;
double energy(const Matrix3& info,const Vector3& r){Big sum=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j)sum+=Big(r[i])*Big(info(i,j))*Big(r[j]);return double(sum/2);}
int main(){try {
  for(int exponent:{-400,0,400}){
    const double scale=std::ldexp(1.,exponent),a=(std::ldexp(1.,53)+4.)*scale,b=std::ldexp(1.,53)*scale;
    Matrix3 info;info<<a,b,0,b,a,0,0,0,scale;
    precise::Whitening w(info);const Vector3 r(1,-1,0),weighted=w.vector(r);
    check(std::abs(.5*weighted.squaredNorm()/scale-4.)<2e-14,"exact weak-direction energy");
    check((precise::original_product(info,r)/scale-Vector3(4,-4,0)).norm()<1e-14,"original weighted gradient");
    check(precise::needs_precision(info),"small pivot detection");
  }
  Matrix3 info;info<<2693538350855.096,-157146640359.0913,0,-157146640359.0913,9168262482.076782,0,0,0,636.440966;
  const precise::Whitening w(info);const Vector3 r(-0.0008459986500432724,-0.01450065877031801,-0.00031608221007084497);
  check(std::abs(.5*w.vector(r).squaredNorm()-energy(info,r))<3e-18,"recorded original energy");
  for(const Matrix3& invalid:{Matrix3::Ones().eval(),(Matrix3()<<1,2,0,2,1,0,0,0,1).finished()}){
    bool failed=false;try{precise::Whitening bad(invalid);}catch(const std::runtime_error&){failed=true;}check(failed,"nonpositive information must fail");
  }
  check(!precise::needs_precision(Matrix3::Identity()),"normal weights unchanged");
  bool overflow=false;try{w.vector(Vector3::Constant(std::numeric_limits<double>::max()));}catch(const std::runtime_error&){overflow=true;}check(overflow,"overflow must fail");
  const Pose2 a(2,-1,.4),b(3,2,-.2),z(1,.1,.3);
  precise::Between f(1,2,z,w);Matrix ja,jb;
  const Vector residual=f.evaluateError(a,b,ja,jb);
  BetweenFactor<Pose2> raw(1,2,z,noiseModel::Unit::Create(3));
  check(std::abs(.5*residual.squaredNorm()-energy(info,raw.evaluateError(a,b)))<2e-15*std::max(1.,energy(info,raw.evaluateError(a,b))),"factor original energy");
  for(int c=0;c<6;++c){Vector3 d=Vector3::Zero();d[c%3]=1e-5;
    const Vector plus=c<3?f.evaluateError(a.retract(d),b):f.evaluateError(a,b.retract(d));
    const Vector minus=c<3?f.evaluateError(a.retract(-d),b):f.evaluateError(a,b.retract(-d));
    const Vector derivative=c<3?Vector(ja.col(c)):Vector(jb.col(c-3));
    check((derivative-(plus-minus)/2e-5).norm()<2e-8*std::max(1.,derivative.norm()),"weighted right-chart Jacobian");
  }
  precise::RootBetween root(2,a,z,info);Matrix h;check((root.evaluateError(b,h)-residual).norm()<1e-12,"root conditioning residual");check((h-jb).norm()<1e-12,"root conditioning Jacobian");
  Values values;values.insert(1,a);values.insert(2,b);
  check(f.equals(*f.clone())&&root.equals(*root.clone()),"factor clone identity");
  check(f.error(values)==f.clone()->error(values)&&root.error(values)==root.clone()->error(values),"clones preserve whitening");
  Matrix3 normal;normal<<400,30,10,30,500,-15,10,-15,2500;
  precise::RootBetween normal_root(2,a,z,normal);
  check(std::abs(normal_root.error(values)-energy(normal,raw.evaluateError(a,b)))<1e-12,"normal root original energy");
  std::cout<<"exact scaled eigenpair, 200-digit original energy, Jacobians, conditioned root, clone, invalid pivots and overflow pass\n";
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
