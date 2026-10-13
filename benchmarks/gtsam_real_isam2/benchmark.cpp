// External native iSAM2 accuracy-matched reference. No Rust FFI/runtime dependency.
#include <gtsam/config.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#ifndef GTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR
#error "Exact BetweenFactor Jacobians required"
#endif
#ifndef NDEBUG
#error "Release required"
#endif
using namespace gtsam;
namespace fs=std::filesystem;
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
double derivative_probe() {
  const Pose2 a(2,-1,.4),b(3,2,-.2),z(1,.1,.3);
  BetweenFactor<Pose2> f(1,2,z,noiseModel::Unit::Create(3)); Matrix ja,jb;
  const auto p=z.inverse().compose(a.between(b));
  require((f.evaluateError(a,b,ja,jb)-Vector3(p.x(),p.y(),p.theta())).norm()<1e-12,"residual coordinate probe failed");
  double maximum=0.;
  for(int c=0;c<6;++c) {
    Vector3 d=Vector3::Zero(); d[c%3]=1e-6;
    const Vector plus=c<3?f.evaluateError(a.retract(d),b):f.evaluateError(a,b.retract(d));
    const Vector minus=c<3?f.evaluateError(a.retract(-d),b):f.evaluateError(a,b.retract(-d));
    const Vector analytical=c<3?Vector(ja.col(c)):Vector(jb.col(c-3));
    maximum=std::max(maximum,(analytical-(plus-minus)/2e-6).cwiseAbs().maxCoeff());
  }
  require(std::isfinite(maximum)&&maximum<1e-7,"Jacobian probe failed"); return maximum;
}
struct Edge {Key a,b; Pose2 z; Matrix3 information;};
struct Frame {Key key; Pose2 seed; std::vector<Edge> edges; NonlinearFactorGraph graph; Values initial;};
Pose2 pose(std::istream& f) {double x,y,t; require(bool(f>>x>>y>>t),"invalid pose"); require(std::isfinite(x)&&std::isfinite(y)&&std::isfinite(t),"nonfinite pose"); return Pose2(x,y,t);}
double gradient(const Values& values,const Pose2& root,const std::vector<Edge>& edges) {
  std::map<Key,Vector3> g, diagonal;
  for(const auto& e:edges) {
    auto a=e.a==0?root:values.at<Pose2>(e.a); const auto& b=values.at<Pose2>(e.b);
    // Original information energy, independent of the whitened factor graph.
    const auto relative=a.between(b); const auto rpose=e.z.inverse().compose(relative);
    const Vector3 r(rpose.x(),rpose.y(),rpose.theta()); const Vector3 w=e.information*r;
    const double cz=std::cos(e.z.theta()),sz=std::sin(e.z.theta());
    const double qx=relative.x(),qy=relative.y();
    Matrix3 ja,jb;
    ja<<-cz,-sz,cz*qy-sz*qx,sz,-cz,-sz*qy-cz*qx,0,0,-1;
    const double theta=b.theta()-a.theta()-e.z.theta();
    jb<<std::cos(theta),-std::sin(theta),0,std::sin(theta),std::cos(theta),0,0,0,1;
    if(e.a!=0) {if(!g.count(e.a)) {g[e.a]=Vector3::Zero(); diagonal[e.a]=Vector3::Zero();} g[e.a]+=ja.transpose()*w; diagonal[e.a]+=(ja.transpose()*e.information*ja).diagonal();}
    if(!g.count(e.b)) {g[e.b]=Vector3::Zero(); diagonal[e.b]=Vector3::Zero();}
    g[e.b]+=jb.transpose()*w; diagonal[e.b]+=(jb.transpose()*e.information*jb).diagonal();
  }
  double maximum=0.; for(const auto& entry:g) {for(int c=0;c<3;++c) {const double scale=std::sqrt(diagonal.at(entry.first)[c]); require(std::isfinite(scale)&&scale>0.,"invalid column norm"); maximum=std::max(maximum,std::abs(entry.second[c])/scale);}}
  require(std::isfinite(maximum),"nonfinite gradient"); return maximum;
}
int main(int argc,char** argv) {try {
  require(argc==5,"usage: gtsam_isam2 input NEW_DIRECTORY repeats warmup");
  require(argv[3][0]!='-'&&argv[4][0]!='-',"negative repetition count");
  const size_t repeats=std::stoull(argv[3]),warmup=std::stoull(argv[4]);
  require(repeats>0&&warmup<=SIZE_MAX-repeats,"invalid repetitions");
  const auto discrepancy=derivative_probe();
  std::ifstream input(argv[1]); require(input.good(),"missing input"); std::string tag;
  require(bool(input>>tag)&&tag=="ROOT","missing ROOT"); const auto root=pose(input);
  std::vector<Frame> frames;
  while(input>>tag) {
    require(tag=="FRAME","missing FRAME"); Key key; require(bool(input>>key)&&key==frames.size()+1,"invalid key");
    const auto seed=pose(input); size_t count; require(bool(input>>count)&&count>0&&count<=10000,"invalid edge count");
    Frame frame{key,seed,{},{},{}}; frame.initial.insert(key,seed);
    for(size_t i=0;i<count;++i) {
      Key a,b; require(bool(input>>tag>>a>>b)&&tag=="EDGE"&&a<b&&b<=key,"invalid edge");
      const auto z=pose(input); double w[6]; for(auto& v:w) {require(bool(input>>v)&&std::isfinite(v),"invalid information");}
      Matrix3 information; information<<w[0],w[1],w[2],w[1],w[3],w[4],w[2],w[4],w[5];
      require(information.llt().info()==Eigen::Success,"non SPD information");
      const auto noise=noiseModel::Gaussian::Information(information);
      frame.edges.push_back(Edge{a,b,z,information});
      if(a==0) frame.graph.emplace_shared<PriorFactor<Pose2>>(b,root.compose(z),noise);
      else frame.graph.emplace_shared<BetweenFactor<Pose2>>(a,b,z,noise);
    }
    frames.push_back(frame);
  }
  require(!frames.empty()&&frames.size()<=10000,"invalid frame count");
  const fs::path out(argv[2]); require(fs::create_directory(out),"output must be new");
  ISAM2Params params; params.relinearizeThreshold=0.; params.relinearizeSkip=1;
  params.factorization=ISAM2Params::QR; params.optimizationParams=ISAM2GaussNewtonParams(0.);
  params.enablePartialRelinearizationCheck=false; params.cacheLinearizedFactors=true;
  std::ofstream options(out/"options.txt"); auto* stream=std::cout.rdbuf(options.rdbuf()); params.print(); std::cout.rdbuf(stream);
  options<<"exact_between=ON\nroot eliminated with equivalent prior\ngradient_metric=weighted-column-normalized\nadditional empty updates until original gradient<=8e-8, maximum 200 calls\nversion="<<GTSAM_VERSION_STRING<<"\n";
  options<<std::setprecision(17)<<"jacobian_probe="<<discrepancy<<'\n';
  std::ofstream samples(out/"samples.csv"),updates(out/"updates.csv"),poses(out/"poses.csv");
  samples<<"sample,elapsed_ns,updates,allocations,requested_bytes\n";
  updates<<"sample,frame,elapsed_ns,iterations,cost,entries,allocations,requested_bytes\n"<<std::setprecision(17);
  poses<<"frame,key,x,y,angle\n"<<std::setprecision(17);
  for(size_t round=0;round<warmup+repeats;++round) {
    ISAM2 isam(params); std::vector<Edge> edges; NonlinearFactorGraph fullgraph;
    long long elapsed=0;
    for(const auto& frame:frames) {
      edges.insert(edges.end(),frame.edges.begin(),frame.edges.end()); fullgraph.push_back(frame.graph);
      const auto start=std::chrono::steady_clock::now();
      isam.update(frame.graph,frame.initial); Values values=isam.calculateBestEstimate(); size_t calls=1;
      while(gradient(values,root,edges)>8e-8&&calls<200) {isam.update(); values=isam.calculateBestEstimate(); ++calls;}
      require(gradient(values,root,edges)<=8e-8,"iSAM2 did not meet stationarity target");
      const double cost=fullgraph.error(values);
      // Build a complete owned report (including the conditioned root), as ropt does.
      std::map<Key,Pose2> report=values.extract<Pose2>(); report.emplace(0,root);
      const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count(); elapsed+=ns;
      if(round>=warmup) updates<<round-warmup<<','<<frame.key<<','<<ns<<','<<calls<<','<<cost<<",0,0,0\n";
      if(round==warmup) for(const auto& entry:report) {const auto& p=entry.second; poses<<frame.key<<','<<entry.first<<','<<p.x()<<','<<p.y()<<','<<p.theta()<<'\n';}
    }
    if(round>=warmup) samples<<round-warmup<<','<<elapsed<<','<<frames.size()<<",0,0\n";
  }
  require(samples.good()&&updates.good()&&poses.good()&&options.good(),"output failure");
  return 0;
}catch(const std::exception& e) {std::cerr<<e.what()<<'\n'; return 1;}}
