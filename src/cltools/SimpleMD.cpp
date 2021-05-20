/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2012-2020 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "CLTool.h"
#include "CLToolRegister.h"
#include "core/PlumedMain.h"
#include "tools/Vector.h"
#include "tools/Random.h"
#include "tools/OpenMP.h"
#include <string>
#include <cstdio>
#include <cmath>
#include <vector>
#include <memory>

#include <fstream>
#include <iostream>
#include "tools/Angle.h"
#include "tools/Torsion.h"

using namespace std;

namespace PLMD {
namespace cltools {

//+PLUMEDOC TOOLS simplemd
/*
simplemd allows one to do molecular dynamics on systems of Lennard-Jones atoms.

The input to simplemd is specified in an input file. Configurations are input and
output in xyz format. The input file should contain one directive per line.
The directives available are as follows:

\par Examples

You run an MD simulation using simplemd with the following command:
\verbatim
plumed simplemd < in
\endverbatim

The following is an example of an input file for a simplemd calculation. This file
instructs simplemd to do 50 steps of MD at a temperature of 0.722
\verbatim
nputfile input.xyz
outputfile output.xyz
temperature 0.722
tstep 0.005
friction 1
forcecutoff 2.5
listcutoff  3.0
nstep 50
nconfig 10 trajectory.xyz
nstat   10 energies.dat
\endverbatim

If you run the following a description of all the directives that can be used in the
input file will be output.
\verbatim
plumed simplemd --help
\endverbatim

*/
//+ENDPLUMEDOC

class SimpleMD:
  public PLMD::CLTool
{
  string description()const override {
    return "run lj code";
  }

  bool write_positions_first;
  bool write_statistics_first;
  int write_statistics_last_time_reopened;
  FILE* write_statistics_fp;

  vector<int> bonds_1, bonds_2;
  vector<double> bonds_ref, bonds_kappa;
  vector<int> angles_1, angles_2, angles_3;
  vector<double> angles_ref, angles_kappa;
  vector<int> torsions_1, torsions_2, torsions_3, torsions_4;
  vector<double> torsions_ref, torsions_kappa;
  vector<int> pairs_1, pairs_2;
  vector<double> pairs_ref, pairs_kappa;

public:
  static void registerKeywords( Keywords& keys ) {
    keys.add("compulsory","nstep","The number of steps of dynamics you want to run");
    keys.add("compulsory","temperature","NVE","the temperature at which you wish to run the simulation in LJ units");
    keys.add("compulsory","friction","off","The friction (in LJ units) for the Langevin thermostat that is used to keep the temperature constant");
    keys.add("compulsory","tstep","0.005","the integration timestep in LJ units");
    keys.add("compulsory","epsilon","1.0","LJ parameter");
    keys.add("compulsory","sigma","1.0","LJ parameter");
    keys.add("compulsory","inputfile","An xyz file containing the initial configuration of the system");
    keys.add("compulsory","forcecutoff","2.5","");
    keys.add("compulsory","listcutoff","3.0","");
    keys.add("compulsory","outputfile","An output xyz file containing the final configuration of the system");
    keys.add("compulsory","nconfig","10","The frequency with which to write configurations to the trajectory file followed by the name of the trajectory file");
    keys.add("compulsory","nstat","1","The frequency with which to write the statistics to the statistics file followed by the name of the statistics file");
    keys.add("compulsory","maxneighbours","10000","The maximum number of neighbors an atom can have");
    keys.add("compulsory","idum","0","The random number seed");
    keys.add("compulsory","ndim","3","The dimensionality of the system (some interesting LJ clusters are two dimensional)");
    keys.add("compulsory","wrapatoms","false","If true, atomic coordinates are written wrapped in minimal cell");
    keys.add("optional","bonds_file","file containing bonds data");
    keys.add("optional","angles_file","file containing angles data");
    keys.add("optional","torsions_file","file containing torsions data");
    keys.add("optional","pairs_file","file containing pairs data");
  }

  explicit SimpleMD( const CLToolOptions& co ) :
    CLTool(co),
    write_positions_first(true),
    write_statistics_first(true),
    write_statistics_last_time_reopened(0),
    write_statistics_fp(NULL)
  {
    inputdata=ifile;
  }

private:

  void
  read_input(double& temperature,
             double& tstep,
             double& friction,
             double& forcecutoff,
             double& listcutoff,
             int&    nstep,
             int&    nconfig,
             int&    nstat,
             bool&   wrapatoms,
             string& inputfile,
             string& outputfile,
             string& trajfile,
             string& statfile,
             int&    maxneighbours,
             int&    ndim,
             int&    idum,
             double& epsilon,
             double& sigma,
             string& bonds_file,
             string& angles_file,
             string& torsions_file,
             string& pairs_file)
  {

    // Read everything from input file
    char buffer1[256];
    std::string tempstr; parse("temperature",tempstr);
    if( tempstr!="NVE" ) Tools::convert(tempstr,temperature);
    parse("tstep",tstep);
    std::string frictionstr; parse("friction",frictionstr);
    if( tempstr!="NVE" ) {
      if(frictionstr=="off") { fprintf(stderr,"Specify friction for thermostat\n"); exit(1); }
      Tools::convert(frictionstr,friction);
    }
    parse("forcecutoff",forcecutoff);
    parse("listcutoff",listcutoff);
    parse("nstep",nstep);
    parse("maxneighbours",maxneighbours);
    parse("idum",idum);
    parse("epsilon",epsilon);
    parse("sigma",sigma);

    // Read in stuff with sanity checks
    parse("inputfile",inputfile);
    if(inputfile.length()==0) {
      fprintf(stderr,"Specify input file\n");
      exit(1);
    }
    parse("outputfile",outputfile);
    if(outputfile.length()==0) {
      fprintf(stderr,"Specify output file\n");
      exit(1);
    }
    std::string nconfstr; parse("nconfig",nconfstr);
    sscanf(nconfstr.c_str(),"%100d %255s",&nconfig,buffer1);
    trajfile=buffer1;
    if(trajfile.length()==0) {
      fprintf(stderr,"Specify traj file\n");
      exit(1);
    }
    std::string nstatstr; parse("nstat",nstatstr);
    sscanf(nstatstr.c_str(),"%100d %255s",&nstat,buffer1);
    statfile=buffer1;
    if(statfile.length()==0) {
      fprintf(stderr,"Specify stat file\n");
      exit(1);
    }
    parse("ndim",ndim);
    if(ndim<1 || ndim>3) {
      fprintf(stderr,"ndim should be 1,2 or 3\n");
      exit(1);
    }
    std::string w;
    parse("wrapatoms",w);
    wrapatoms=false;
    if(w.length()>0 && (w[0]=='T' || w[0]=='t')) wrapatoms=true;
    
    parse("bonds_file", bonds_file);
    parse("angles_file", angles_file);
    parse("torsions_file", torsions_file);
    parse("pairs_file", pairs_file);
  }

  void read_natoms(const string & inputfile,int & natoms) {
// read the number of atoms in file "input.xyz"
    FILE* fp=fopen(inputfile.c_str(),"r");
    if(!fp) {
      fprintf(stderr,"ERROR: file %s not found\n",inputfile.c_str());
      exit(1);
    }

// call fclose when fp goes out of scope
    auto deleter=[](FILE* f) { fclose(f); };
    std::unique_ptr<FILE,decltype(deleter)> fp_deleter(fp,deleter);

    int ret=fscanf(fp,"%1000d",&natoms);
    if(ret==0) plumed_error() <<"Error reading number of atoms from file "<<inputfile;
  }

  void read_positions(const string& inputfile,int natoms,vector<Vector>& positions,double cell[3]) {
// read positions and cell from a file called inputfile
// natoms (input variable) and number of atoms in the file should be consistent
    FILE* fp=fopen(inputfile.c_str(),"r");
    if(!fp) {
      fprintf(stderr,"ERROR: file %s not found\n",inputfile.c_str());
      exit(1);
    }
// call fclose when fp goes out of scope
    auto deleter=[](FILE* f) { fclose(f); };
    std::unique_ptr<FILE,decltype(deleter)> fp_deleter(fp,deleter);

    char buffer[256];
    char atomname[256];
    char* cret=fgets(buffer,256,fp);
    if(cret==nullptr) plumed_error() <<"Error reading buffer from file "<<inputfile;
    int ret=fscanf(fp,"%1000lf %1000lf %1000lf",&cell[0],&cell[1],&cell[2]);
    if(ret==0) plumed_error() <<"Error reading cell line from file "<<inputfile;
    for(int i=0; i<natoms; i++) {
      ret=fscanf(fp,"%255s %1000lf %1000lf %1000lf",atomname,&positions[i][0],&positions[i][1],&positions[i][2]);
// note: atomname is read but not used
      if(ret==0) plumed_error() <<"Error reading atom line from file "<<inputfile;
    }
  }

  void randomize_velocities(const int natoms,const int ndim,const double temperature,const vector<double>&masses,vector<Vector>& velocities,Random&random) {
// randomize the velocities according to the temperature
    for(int iatom=0; iatom<natoms; iatom++) for(int i=0; i<ndim; i++)
        velocities[iatom][i]=sqrt(temperature/masses[iatom])*random.Gaussian();
  }

  void pbc(const double cell[3],const Vector & vin,Vector & vout) {
// apply periodic boundary condition to a vector
    for(int i=0; i<3; i++) {
      vout[i]=vin[i]-floor(vin[i]/cell[i]+0.5)*cell[i];
    }
  }

  void check_list(const int natoms,const vector<Vector>& positions,const vector<Vector>&positions0,const double listcutoff,
                  const double forcecutoff,bool & recompute)
  {
// check if the neighbour list have to be recomputed
    recompute=false;
    auto delta2=(0.5*(listcutoff-forcecutoff))*(0.5*(listcutoff-forcecutoff));
// if ANY atom moved more than half of the skin thickness, recompute is set to .true.
    for(int iatom=0; iatom<natoms; iatom++) {
      if(modulo2(positions[iatom]-positions0[iatom])>delta2) recompute=true;
    }
  }


  void compute_list(const int natoms,const vector<Vector>& positions,const double cell[3],const double listcutoff,
                    vector<vector<int> >& list) {
    double listcutoff2=listcutoff*listcutoff; // squared list cutoff
    list.assign(natoms,vector<int>());
#   pragma omp parallel for num_threads(OpenMP::getNumThreads()) schedule(static,1)
    for(int iatom=0; iatom<natoms-1; iatom++) {
      for(int jatom=iatom+1; jatom<natoms; jatom++) {
        auto distance=positions[iatom]-positions[jatom];
        Vector distance_pbc; // minimum-image distance of the two atoms
        pbc(cell,distance,distance_pbc);
// if the interparticle distance is larger than the cutoff, skip
        if(modulo2(distance_pbc)>listcutoff2)continue;
        list[iatom].push_back(jatom);
      }
    }
  }

  void compute_forces(const int natoms,double epsilon, double sigma,const vector<Vector>& positions,const double cell[3],
                      double forcecutoff,const vector<vector<int> >& list,vector<Vector>& forces,double & engconf)
  {
    double forcecutoff2=forcecutoff*forcecutoff; // squared force cutoff
    engconf=0.0;
    for(int i=0; i<natoms; i++)for(int k=0; k<3; k++) forces[i][k]=0.0;
    double engcorrection=4.0*epsilon*(1.0/pow(forcecutoff2/(sigma*sigma),6.0)-1.0/pow(forcecutoff2/(sigma*sigma),3)); // energy necessary shift the potential avoiding discontinuities
#   pragma omp parallel num_threads(OpenMP::getNumThreads())
    {
      std::vector<Vector> omp_forces(forces.size());
      #pragma omp for reduction(+:engconf) schedule(static,1) nowait
      for(int iatom=0; iatom<natoms-1; iatom++) {
        for(int jlist=0;jlist<list[iatom].size();jlist++){
          const int jatom=list[iatom][jlist];
          auto distance=positions[iatom]-positions[jatom];
          Vector distance_pbc;    // minimum-image distance of the two atoms
          pbc(cell,distance,distance_pbc);
          auto distance_pbc2=modulo2(distance_pbc);   // squared minimum-image distance
// if the interparticle distance is larger than the cutoff, skip
          if(distance_pbc2>forcecutoff2) continue;
          auto distance_pbcm2=sigma*sigma/distance_pbc2;
          auto distance_pbcm6=distance_pbcm2*distance_pbcm2*distance_pbcm2;
          auto distance_pbcm8=distance_pbcm6*distance_pbcm2;
          auto distance_pbcm12=distance_pbcm6*distance_pbcm6;
          auto distance_pbcm14=distance_pbcm12*distance_pbcm2;
          engconf+=4.0*epsilon*(distance_pbcm12 - distance_pbcm6) - engcorrection;
          auto f=24.0*distance_pbc*(2.0*distance_pbcm14-distance_pbcm8)*epsilon/sigma;
          omp_forces[iatom]+=f;
          omp_forces[jatom]-=f;
        }
      }

      // Bonds contribution
      #pragma omp for reduction(+:engconf) schedule(static,1) nowait
      for(int i=0; i<bonds_1.size(); ++i){
        int iatom = bonds_1[i];
        int jatom = bonds_2[i];
        double ref = bonds_ref[i];
        double kappa = bonds_kappa[i];
        
        auto distance = positions[iatom] - positions[jatom];
        Vector distance_pbc;
        pbc(cell,distance,distance_pbc); // distance_pbc = \vec{r_2}-\vec{r_1}
        auto distance_pbc_mod=modulo(distance_pbc);   // r = |\vec{r_2}-\vec{r_1}|
        auto delta = distance_pbc_mod - ref; // r-d_0
        
        engconf += 0.5*kappa*delta*delta;
        auto f = kappa*delta*distance_pbc/distance_pbc_mod;
        
        omp_forces[iatom]-=f;
        omp_forces[jatom]+=f;
      }// for bonds contribution

      // Angles contribution
      #pragma omp for reduction(+:engconf) schedule(static,1) nowait
      for(int i=0; i<angles_1.size(); ++i){
        int iatom = angles_1[i];
        int jatom = angles_2[i];
        int katom = angles_3[i];
        double ref = angles_ref[i];
        double kappa = angles_kappa[i];

        auto r21 = positions[jatom] - positions[iatom];
        auto r23 = positions[jatom] - positions[katom];
        Vector r21_pbc, r23_pbc, dr21, dr23;
        pbc(cell,r21,r21_pbc);
        pbc(cell,r23,r23_pbc);
        
        // See plumed2/src/multicolvar/Angles.{cpp,h}
        Angle a;
        double angle = a.compute(r21_pbc, r23_pbc, dr21, dr23);
        auto delta = angle -ref;
        
        engconf += 0.5*kappa*delta*delta;
        auto f21 = kappa*delta*dr21;
        auto f23 = kappa*delta*dr23;
        
        omp_forces[iatom] += f21;
        omp_forces[jatom] -= (f23 + f21);
        omp_forces[katom] += f23;
      }// for angles contribution

      // Torsions contribution
      #pragma omp for reduction(+:engconf) schedule(static,1) nowait
      for(int i=0; i<torsions_1.size(); ++i){
        int iatom = torsions_1[i];
        int jatom = torsions_2[i];
        int katom = torsions_3[i];
        int latom = torsions_4[i];
        double ref = torsions_ref[i];
        double kappa = torsions_kappa[i];

        auto r12 = positions[jatom] - positions[iatom];
        auto r23 = positions[katom] - positions[jatom];
        auto r34 = positions[latom] - positions[katom];
        Vector r12_pbc, r23_pbc, r34_pbc, dr12, dr23, dr34;
        pbc(cell,-r12,r12_pbc);
        pbc(cell,-r23,r23_pbc);
        pbc(cell,-r34,r34_pbc);

        // See plumed2/src/multicolvar/Torsions.{cpp,h}
        Torsion t;
        // PBC changes nothing - verified
        double angle = t.compute(r12_pbc, r23_pbc, r34_pbc, dr12, dr23, dr34);
        angle = t.compute(r12_pbc, r23_pbc, r34_pbc);
        // double angle = t.compute(-r12, -r23, -r34, dr12, dr23, dr34);
        // angle = t.compute(-r12, -r23, -r34);

        auto delta = angle - ref;

        // engconf += 0.5*kappa*(delta)*(delta);
        // auto f12 = kappa*(delta)*dr12;
        // auto f23 = kappa*(delta)*dr23;
        // auto f34 = kappa*(delta)*dr34;

        // See: plumed2/src/multicolvar/DihedralCorrelation.cpp AND plumed2/src/multicolvar/AlphaBeta.cpp
        engconf += kappa*(1.0+cos(delta));
        auto f12 = -kappa*sin(delta)*dr12;
        auto f23 = -kappa*sin(delta)*dr23;
        auto f34 = -kappa*sin(delta)*dr34;

        omp_forces[iatom] += f12;
        omp_forces[jatom] -= (-f23 + f12);
        omp_forces[katom] -= (-f34 + f23);
        omp_forces[latom] -= f34;
      }// for torsions contribution

      // Pairs contribution
      #pragma omp for reduction(+:engconf) schedule(static,1) nowait
      for(int i=0; i<pairs_1.size(); ++i){
        int iatom = pairs_1[i];
        int jatom = pairs_2[i];
        double ref = pairs_ref[i];
        double kappa = pairs_kappa[i];
        
        auto distance = positions[jatom] - positions[iatom];
        /* Remember the no pbc flag in the pairs.dat */
        // Vector distance_pbc;
        // pbc(cell,distance,distance_pbc); // distance_pbc = \vec{r_2}-\vec{r_1}
        auto distance_mod=modulo(distance);   // r = |\vec{r_2}-\vec{r_1}|
        auto delta = distance_mod - ref; // x-r
        auto delta5 = pow(delta,5);
        auto delta6 = delta5*delta;

        // Reference for the step function used in pairs.dat: plumed2/src/function/Custom.cpp
        engconf += (delta > 0.0 ? -kappa/(1.0+delta6) : -kappa);
        auto f = (delta >0.0 ? 6.0*kappa*delta5/(delta6+1.0)/(delta6+1.0) : 0.0)*distance/distance_mod;
        
        omp_forces[iatom]+=f;
        omp_forces[jatom]-=f;
      }// for pairs contribution


#     pragma omp critical
      for(unsigned i=0; i<omp_forces.size(); i++) forces[i]+=omp_forces[i];
    }

  };

  void compute_engkin(const int natoms,const vector<double>& masses,const vector<Vector>& velocities,double & engkin)
  {
// calculate the kinetic energy from the velocities
    engkin=0.0;
    for(int iatom=0; iatom<natoms; iatom++) {
        engkin+=0.5*masses[iatom]*modulo2(velocities[iatom]);
      }
  }


  void thermostat(const int natoms,const int ndim,const vector<double>& masses,const double dt,const double friction,
                  const double temperature,vector<Vector>& velocities,double & engint,Random & random) {
// Langevin thermostat, implemented as described in Bussi and Parrinello, Phys. Rev. E (2007)
// it is a linear combination of old velocities and new, randomly chosen, velocity,
// with proper coefficients
    double c1=exp(-friction*dt);
    for(int iatom=0; iatom<natoms; iatom++) {
      double c2=sqrt((1.0-c1*c1)*temperature/masses[iatom]);
      for(int i=0; i<ndim; i++) {
        engint+=0.5*masses[iatom]*velocities[iatom][i]*velocities[iatom][i];
        velocities[iatom][i]=c1*velocities[iatom][i]+c2*random.Gaussian();
        engint-=0.5*masses[iatom]*velocities[iatom][i]*velocities[iatom][i];
      }
    }
  }

  void write_positions(const string& trajfile,int natoms,const vector<Vector>& positions,const double cell[3],const bool wrapatoms)
  {
// write positions on file trajfile
// positions are appended at the end of the file
    Vector pos;
    FILE*fp;
    if(write_positions_first) {
      fp=fopen(trajfile.c_str(),"w");
      write_positions_first=false;
    } else {
      fp=fopen(trajfile.c_str(),"a");
    }
    fprintf(fp,"%d\n",natoms);
    fprintf(fp,"%f %f %f\n",cell[0],cell[1],cell[2]);
    for(int iatom=0; iatom<natoms; iatom++) {
// usually, it is better not to apply pbc here, so that diffusion
// is more easily calculated from a trajectory file:
      if(wrapatoms) pbc(cell,positions[iatom],pos);
      else pos=positions[iatom];
      fprintf(fp,"Ar %10.7f %10.7f %10.7f\n",pos[0],pos[1],pos[2]);
    }
    fclose(fp);
  }

  void write_final_positions(const string& outputfile,int natoms,const vector<Vector>& positions,const double cell[3],const bool wrapatoms)
  {
// write positions on file outputfile
    Vector pos;
    FILE*fp;
    fp=fopen(outputfile.c_str(),"w");
    fprintf(fp,"%d\n",natoms);
    fprintf(fp,"%f %f %f\n",cell[0],cell[1],cell[2]);
    for(int iatom=0; iatom<natoms; iatom++) {
// usually, it is better not to apply pbc here, so that diffusion
// is more easily calculated from a trajectory file:
      if(wrapatoms) pbc(cell,positions[iatom],pos);
      else pos=positions[iatom];
      fprintf(fp,"Ar %10.7f %10.7f %10.7f\n",pos[0],pos[1],pos[2]);
    }
    fclose(fp);
  }


  void write_statistics(const string & statfile,const int istep,const double tstep,
                        const int natoms,const int ndim,const double engkin,const double engconf,const double engint) {
// write statistics on file statfile
    if(write_statistics_first) {
// first time this routine is called, open the file
      write_statistics_fp=fopen(statfile.c_str(),"w");
      write_statistics_first=false;
    }
    if(istep-write_statistics_last_time_reopened>100) {
// every 100 steps, reopen the file to flush the buffer
      fclose(write_statistics_fp);
      write_statistics_fp=fopen(statfile.c_str(),"a");
      write_statistics_last_time_reopened=istep;
    }
    fprintf(write_statistics_fp,"%d %f %f %f %f %f\n",istep,istep*tstep,2.0*engkin/double(ndim*natoms),engconf,engkin+engconf,engkin+engconf+engint);
  }



  int main(FILE* in,FILE*out,PLMD::Communicator& pc) override {
    int            natoms;       // number of atoms
    vector<Vector> positions;    // atomic positions
    vector<Vector> velocities;   // velocities
    vector<double> masses;       // masses
    vector<Vector> forces;       // forces
    double         cell[3];      // cell size
    double         cell9[3][3];  // cell size

// neighbour list variables
// see Allen and Tildesey book for details
    vector< vector<int> >  list; // neighbour list
    vector<Vector> positions0;   // reference atomic positions, i.e. positions when the neighbour list

// input parameters
// all of them have a reasonable default value, set in read_input()
    double      tstep;             // simulation timestep
    double      temperature;       // temperature
    double      friction;          // friction for Langevin dynamics (for NVE, use 0)
    double      listcutoff;        // cutoff for neighbour list
    double      forcecutoff;       // cutoff for forces
    int         nstep;             // number of steps
    int         nconfig;           // stride for output of configurations
    int         nstat;             // stride for output of statistics
    int         maxneighbour;      // maximum average number of neighbours per atom
    int         ndim;              // dimensionality of the system (1, 2, or 3)
    int         idum;              // seed
    int         plumedWantsToStop; // stop flag
    bool        wrapatoms;         // if true, atomic coordinates are written wrapped in minimal cell
    string      inputfile;         // name of file with starting configuration (xyz)
    string      outputfile;        // name of file with final configuration (xyz)
    string      trajfile;          // name of the trajectory file (xyz)
    string      statfile;          // name of the file with statistics

    double      epsilon, sigma;    // LJ parameters

    double engkin;                 // kinetic energy
    double engconf;                // configurational energy
    double engint;                 // integral for conserved energy in Langevin dynamics

    bool recompute_list;           // control if the neighbour list have to be recomputed

    Random random;                 // random numbers stream

    string bonds_file;             // name of file containing bonds data
    string angles_file;            // name of file containing angles data
    string torsions_file;          // name of file containing torsions data
    string pairs_file;             // name of file containing pairs data

    std::unique_ptr<PlumedMain> plumed;

// Commenting the next line it is possible to switch-off plumed
    plumed.reset(new PLMD::PlumedMain);

    if(plumed) {
      int s=sizeof(double);
      plumed->cmd("setRealPrecision",&s);
    }

    read_input(temperature,tstep,friction,forcecutoff,
               listcutoff,nstep,nconfig,nstat,
               wrapatoms,inputfile,outputfile,trajfile,statfile,
               maxneighbour,ndim,idum,epsilon,sigma, bonds_file,
               angles_file, torsions_file, pairs_file);

// number of atoms is read from file inputfile
    read_natoms(inputfile,natoms);

// write the parameters in output so they can be checked
    fprintf(out,"%s %s\n","Starting configuration           :",inputfile.c_str());
    fprintf(out,"%s %s\n","Final configuration              :",outputfile.c_str());
    fprintf(out,"%s %d\n","Number of atoms                  :",natoms);
    fprintf(out,"%s %f\n","Temperature                      :",temperature);
    fprintf(out,"%s %f\n","Time step                        :",tstep);
    fprintf(out,"%s %f\n","Friction                         :",friction);
    fprintf(out,"%s %f\n","Cutoff for forces                :",forcecutoff);
    fprintf(out,"%s %f\n","Cutoff for neighbour list        :",listcutoff);
    fprintf(out,"%s %d\n","Number of steps                  :",nstep);
    fprintf(out,"%s %d\n","Stride for trajectory            :",nconfig);
    fprintf(out,"%s %s\n","Trajectory file                  :",trajfile.c_str());
    fprintf(out,"%s %d\n","Stride for statistics            :",nstat);
    fprintf(out,"%s %s\n","Statistics file                  :",statfile.c_str());
    fprintf(out,"%s %d\n","Max average number of neighbours :",maxneighbour);
    fprintf(out,"%s %d\n","Dimensionality                   :",ndim);
    fprintf(out,"%s %d\n","Seed                             :",idum);
    fprintf(out,"%s %s\n","Are atoms wrapped on output?     :",(wrapatoms?"T":"F"));
    fprintf(out,"%s %f\n","Epsilon                          :",epsilon);
    fprintf(out,"%s %f\n","Sigma                            :",sigma);
    fprintf(out,"%s %s\n","bonds file                       :",bonds_file.c_str());
    fprintf(out,"%s %s\n","angles file                      :",angles_file.c_str());
    fprintf(out,"%s %s\n","torsions file                    :",torsions_file.c_str());
    fprintf(out,"%s %s\n","pairs file                       :",pairs_file.c_str());

    // Reading the bonds characteristics
    if(bonds_file.length()!=0) {

      cout << "Reading bonds characteristics..." << endl;
      fstream datfile(bonds_file);
      string line, str;
      // while(!datfile.eof()){
      while(getline(datfile,line)){
        int i,j,n;
        double r, k;
        stringstream sstrm(line);        
        sscanf(line.c_str(), "bond%d: DISTANCE ATOMS=%d,%d", &n, &i, &j);
        getline(datfile, line);
        stringstream sstrm2(line);
        sscanf(line.c_str(), "RESTRAINT ARG=bond%d AT=%lf KAPPA=%lf", &n, &r, &k);
        bonds_1.push_back(i-1);
        bonds_2.push_back(j-1);
        bonds_ref.push_back(r);
        bonds_kappa.push_back(k);
      }
      datfile.close();
      // bond9: DISTANCE ATOMS=10,9
      // RESTRAINT ARG=bond9 AT=5.358068943023682 KAPPA=1000.0

      for(int i=0; i<bonds_1.size(); ++i){
        cout << "bonded atoms: " << bonds_1[i] << "-" << bonds_2[i] << " bonds_ref: " << bonds_ref[i]\
        << " kappa: " << bonds_kappa[i] << "\n";
      }//for
    
    }//if bonds_file

    // Reading the angles characteristics
    if(angles_file.length()!=0) {

      cout << "Reading angles characteristics..." << endl;
      fstream datfile(angles_file);
      string line, str;
      // while(!datfile.eof()){
      while(getline(datfile,line)){
        int i,j,l,n;
        double r, k;
        stringstream sstrm(line);        
        sscanf(line.c_str(), "angle%d: ANGLE ATOMS=%d,%d,%d", &n, &i, &j, &l);
        getline(datfile, line);
        stringstream sstrm2(line);
        sscanf(line.c_str(), "RESTRAINT ARG=angle%d AT=%lf KAPPA=%lf", &n, &r, &k);
        angles_1.push_back(i-1);
        angles_2.push_back(j-1);
        angles_3.push_back(l-1);
        angles_ref.push_back(r);
        angles_kappa.push_back(k);
      }
      datfile.close();
      // angle0: ANGLE ATOMS=3,1,2
      // RESTRAINT ARG=angle0 AT=1.8209148645401 KAPPA=1000.0

      for(int i=0; i<angles_1.size(); ++i){
        cout << "angle-atoms: " << angles_1[i] << "-" << angles_2[i] << "-" << angles_3[i] << " angles_ref: " \
        << angles_ref[i] << " kappa: " << angles_kappa[i] << "\n";
      }//for
    
    }//if angles_file

    // Reading the torsions characteristics
    if(torsions_file.length()!=0) {

      cout << "Reading torsions characteristics..." << endl;
      fstream datfile(torsions_file);
      string line, str;
      // while(!datfile.eof()){
      while(getline(datfile,line)){
        int i,j,l,m,n;
        double r, k;
        stringstream sstrm(line);        
        sscanf(line.c_str(), "torsion%d: TORSION ATOMS=%d,%d,%d,%d", &n, &i, &j, &l, &m);
        getline(datfile, line);
        stringstream sstrm2(line);
        sscanf(line.c_str(), "RESTRAINT ARG=torsion%d AT=%lf KAPPA=%lf", &n, &r, &k);
        torsions_1.push_back(i-1);
        torsions_2.push_back(j-1);
        torsions_3.push_back(l-1);
        torsions_4.push_back(m-1);
        torsions_ref.push_back(r);
        torsions_kappa.push_back(k);
      }
      datfile.close();
      // torsion0: TORSION ATOMS=2,1,3,4
      // RESTRAINT ARG=torsion0 AT=-0.5529302954673767 KAPPA=100.0

      for(int i=0; i<torsions_1.size(); ++i){
        cout << "torsion-atoms: " << torsions_1[i] << "-" << torsions_2[i] << "-" << torsions_3[i] << "-" <<\
        torsions_4[i] << " torsions_ref: " << torsions_ref[i] << " kappa: " << torsions_kappa[i] << "\n";
      }//for
    
    }//if torsions_file

    // Reading the pairs characteristics
    if(pairs_file.length()!=0) {

      cout << "Reading pairs characteristics..." << endl;
      fstream datfile(pairs_file);
      string line, str;
      // while(!datfile.eof()){
      while(getline(datfile,line)){
        int i,j,n;
        double r, k;
        stringstream sstrm(line);        
        sscanf(line.c_str(), "pair%d: DISTANCE ATOMS=%d,%d", &n, &i, &j);
        getline(datfile, line);
        stringstream sstrm2(line);
        sscanf(line.c_str(), "contact%d: CUSTOM ARG=pair%d FUNC=%lf*(-step(%lf-x)-step(x-%*c)/(1+(x-%*c)^6)) PERIODIC=NO", &n, &n, &k, &r);
        pairs_1.push_back(i-1);
        pairs_2.push_back(j-1);
        pairs_ref.push_back(r);
        pairs_kappa.push_back(k);
      }
      datfile.close();
      // pair0: DISTANCE ATOMS=1,8
      // contact0: CUSTOM ARG=pair0 FUNC=20.0*(-step(9.833034470761303-x)-step(x-9.833034470761303)/(1+(x-9.833034470761303)^6)) PERIODIC=NO

      // Removing the last added element
      pairs_1.pop_back();
      pairs_2.pop_back();
      pairs_ref.pop_back();
      pairs_kappa.pop_back();

      for(int i=0; i<pairs_1.size(); ++i){
        cout << "paired atoms: " << pairs_1[i] << "-" << pairs_2[i] << " pairs_ref: " << pairs_ref[i]\
        << " kappa: " << pairs_kappa[i] << "\n";
      }//for
    
    }//if pairs_file

// Setting the seed
    random.setSeed(idum);

// allocation of dynamical arrays
    positions.resize(natoms);
    positions0.resize(natoms);
    velocities.resize(natoms);
    forces.resize(natoms);
    masses.resize(natoms);
    list.resize(natoms);

// masses are hard-coded to 1
    for(int i=0; i<natoms; ++i) masses[i]=1.0;

// energy integral initialized to 0
    engint=0.0;

// positions are read from file inputfile
    read_positions(inputfile,natoms,positions,cell);

// velocities are randomized according to temperature
    randomize_velocities(natoms,ndim,temperature,masses,velocities,random);

    if(plumed) {
      plumed->cmd("setNoVirial");
      plumed->cmd("setNatoms",&natoms);
      plumed->cmd("setMDEngine","simpleMD");
      plumed->cmd("setTimestep",&tstep);
      plumed->cmd("setPlumedDat","plumed.dat");
      int pversion=0;
      plumed->cmd("getApiVersion",&pversion);
// setting kbT is only implemented with api>1
// even if not necessary in principle in SimpleMD (which is part of plumed)
// we leave the check here as a reference
      if(pversion>1) {
        plumed->cmd("setKbT",&temperature);
      }
      plumed->cmd("init");
    }

// neighbour list are computed, and reference positions are saved
    compute_list(natoms,positions,cell,listcutoff,list);

    int list_size=0;
    for(int i=0;i<list.size();i++) list_size+=list[i].size();
    fprintf(out,"List size: %d\n",list_size);
    for(int iatom=0; iatom<natoms; ++iatom) positions0[iatom]=positions[iatom];

// forces are computed before starting md
    compute_forces(natoms,epsilon,sigma,positions,cell,forcecutoff,list,forces,engconf);

// remove forces if ndim<3
    if(ndim<3)
      for(int iatom=0; iatom<natoms; ++iatom) for(int k=ndim; k<3; ++k) forces[iatom][k]=0.0;

// here is the main md loop
// Langevin thermostat is applied before and after a velocity-Verlet integrator
// the overall structure is:
//   thermostat
//   update velocities
//   update positions
//   (eventually recompute neighbour list)
//   compute forces
//   update velocities
//   thermostat
//   (eventually dump output informations)
    for(int istep=0; istep<nstep; istep++) {
      thermostat(natoms,ndim,masses,0.5*tstep,friction,temperature,velocities,engint,random);

      for(int iatom=0; iatom<natoms; iatom++)
          velocities[iatom]+=forces[iatom]*0.5*tstep/masses[iatom];

      for(int iatom=0; iatom<natoms; iatom++)
          positions[iatom]+=velocities[iatom]*tstep;

// a check is performed to decide whether to recalculate the neighbour list
      check_list(natoms,positions,positions0,listcutoff,forcecutoff,recompute_list);
      if(recompute_list) {
        compute_list(natoms,positions,cell,listcutoff,list);
        for(int iatom=0; iatom<natoms; ++iatom) positions0[iatom]=positions[iatom];
        int list_size=0;
        for(int i=0;i<list.size();i++) list_size+=list[i].size();
        fprintf(out,"List size: %d\n",list_size);
      }

      compute_forces(natoms,epsilon,sigma,positions,cell,forcecutoff,list,forces,engconf);

      if(plumed) {
        int istepplusone=istep+1;
        plumedWantsToStop=0;
        for(int i=0; i<3; i++)for(int k=0; k<3; k++) cell9[i][k]=0.0;
        for(int i=0; i<3; i++) cell9[i][i]=cell[i];
        plumed->cmd("setStep",&istepplusone);
        plumed->cmd("setMasses",&masses[0]);
        plumed->cmd("setForces",&forces[0]);
        plumed->cmd("setEnergy",&engconf);
        plumed->cmd("setPositions",&positions[0]);
        plumed->cmd("setBox",cell9);
        plumed->cmd("setStopFlag",&plumedWantsToStop);
        plumed->cmd("calc");
        if(plumedWantsToStop) nstep=istep;
      }
// remove forces if ndim<3
      if(ndim<3)
        for(int iatom=0; iatom<natoms; ++iatom) for(int k=ndim; k<3; ++k) forces[iatom][k]=0.0;

      for(int iatom=0; iatom<natoms; iatom++)
          velocities[iatom]+=forces[iatom]*0.5*tstep/masses[iatom];

      thermostat(natoms,ndim,masses,0.5*tstep,friction,temperature,velocities,engint,random);

// kinetic energy is calculated
      compute_engkin(natoms,masses,velocities,engkin);

// eventually, write positions and statistics
      if((istep+1)%nconfig==0) write_positions(trajfile,natoms,positions,cell,wrapatoms);
      if((istep+1)%nstat==0)   write_statistics(statfile,istep+1,tstep,natoms,ndim,engkin,engconf,engint);

    }

// call final plumed jobs
    plumed->cmd("runFinalJobs");

// write final positions
    write_final_positions(outputfile,natoms,positions,cell,wrapatoms);

// close the statistic file if it was open:
    if(write_statistics_fp) fclose(write_statistics_fp);

    return 0;
  }


};

PLUMED_REGISTER_CLTOOL(SimpleMD,"simplemd")

}
}




