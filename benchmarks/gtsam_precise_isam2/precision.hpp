#pragma once
// Same finite two-component arithmetic as ropt information whitening. Benchmark only.
#include <gtsam/base/Matrix.h>
#include <array>
#include <cmath>
#include <stdexcept>
namespace precise {
struct Wide {
  double hi=0.,lo=0.;
  Wide()=default;
  explicit Wide(double value):hi(value){}
  static Wide sum(double a,double b) {
    Wide out; out.hi=a+b; const double tail=out.hi-a;
    out.lo=(a-(out.hi-tail))+(b-tail); return out;
  }
  Wide add(Wide b) const {const auto s=sum(hi,b.hi); return sum(s.hi,s.lo+lo+b.lo);}
  Wide neg() const {Wide out;out.hi=-hi;out.lo=-lo;return out;}
  Wide sub(Wide b) const {return add(b.neg());}
  Wide mul(Wide b) const {const double high=hi*b.hi; return sum(high,std::fma(hi,b.hi,-high)+hi*b.lo+lo*b.hi+lo*b.lo);}
  Wide div(Wide b) const {
    const Wide first(hi/b.hi),second(sub(b.mul(first)).hi/b.hi);const auto result=first.add(second);
    return result.add(Wide(sub(b.mul(result)).hi/b.hi));
  }
  Wide sqrt() const {const Wide first(std::sqrt(hi));return first.add(sub(first.mul(first)).div(Wide(2.*first.hi)));}
  bool finite() const {return std::isfinite(hi)&&std::isfinite(lo);}
  double value() const {if(!finite())throw std::runtime_error("nonfinite compensated arithmetic");return hi+lo;}
};
inline bool needs_precision(const gtsam::Matrix3& info) {
  Eigen::LLT<gtsam::Matrix3> chol(info);
  if(chol.info()!=Eigen::Success)return true;
  const gtsam::Matrix3 l=chol.matrixL();
  for(int k=0;k<3;++k)if(l(k,k)/std::sqrt(info(k,k))<1./8192.)return true;
  return false;
}
struct Whitening {
  std::array<Wide,9> upper;
  gtsam::Matrix3 information;
  explicit Whitening(const gtsam::Matrix3& info):information(info) {
    std::array<Wide,9> lower;
    for(int row=0;row<3;++row)for(int col=0;col<=row;++col) {
      Wide value(info(row,col));for(int k=0;k<col;++k)value=value.sub(lower[row*3+k].mul(lower[col*3+k]));
      if(!value.finite())throw std::runtime_error("nonfinite Cholesky pivot");
      if(row==col&&value.hi<=0.)throw std::runtime_error("nonpositive information pivot");
      auto entry=row==col?value.sqrt():value.div(lower[col*3+col]);
      if(!entry.finite()){throw std::runtime_error("nonfinite information factor");}
      lower[row*3+col]=entry;
    }
    for(int row=0;row<3;++row)for(int col=row;col<3;++col)upper[row*3+col]=lower[col*3+row];
  }
  gtsam::Vector3 vector(const gtsam::Vector3& x) const {
    gtsam::Vector3 out;for(int row=0;row<3;++row) {
      Wide value;for(int col=row;col<3;++col)value=value.add(upper[row*3+col].mul(Wide(x[col])));out[row]=value.value();
    }return out;
  }
  void matrix(gtsam::Matrix& x) const {
    if(x.rows()!=3)throw std::runtime_error("invalid whitening block rows");
    for(int col=0;col<x.cols();++col){const gtsam::Vector3 raw=x.col(col);x.col(col)=vector(raw);}
  }
};
// Direct original-information products, independent of the square-root factor.
inline gtsam::Vector3 original_product(const gtsam::Matrix3& info,const gtsam::Vector3& r) {
  gtsam::Vector3 out;for(int row=0;row<3;++row) {
    Wide value;for(int col=0;col<3;++col)value=value.add(Wide(info(row,col)).mul(Wide(r[col])));out[row]=value.value();
  }return out;
}
inline double original_column_energy(const gtsam::Matrix3& info,const gtsam::Vector3& x) {
  Wide value;for(int row=0;row<3;++row)for(int col=0;col<3;++col)value=value.add(Wide(x[row]).mul(Wide(info(row,col))).mul(Wide(x[col])));
  return value.value();
}
}
